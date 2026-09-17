"""§8: train a learned dispatch policy for the informal fleet, replacing the
fixed threshold-departure rule, and measure whether its emergent departure
pattern becomes more regular (lower inter-departure variance) as more agents
optimize simultaneously (§9b.3: inter-departure variance vs. population size).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from danfosim.config import Config
from danfosim.demand import n_demand_stops
from danfosim.dispatch_policy import DispatchPolicy, step_learned
from danfosim.network import generate_network
from danfosim.sim import resolve_generator
from danfosim.vehicle import init_fleet


def train_policy(cfg: Config, net, device: torch.device, verbose: bool = True) -> DispatchPolicy:
    policy = DispatchPolicy(hidden=cfg.policy_hidden).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.policy_lr)
    gen = resolve_generator(cfg, device)

    baseline = 0.0
    baseline_decay = 0.99
    fleet = queue = t_hours = None

    for update in range(cfg.policy_steps):
        if update % max(cfg.n_steps, 1) == 0:
            fleet = init_fleet(net, cfg, cfg.n_days, cfg.fleet_size, device)
            queue = torch.zeros(cfg.n_days, n_demand_stops(net), device=device)
            t_hours = cfg.day_start_hour

        fleet, queue, boarded, depart_now, policy_depart, logprob, entropy, reward = step_learned(
            fleet, queue, net, cfg, gen, policy, t_hours
        )
        t_hours += cfg.step_minutes / 60.0

        if policy_depart.any():
            r = reward[policy_depart].detach()
            baseline = baseline_decay * baseline + (1 - baseline_decay) * r.mean().item()
            advantage = r - baseline
            loss = -(advantage * logprob[policy_depart]).mean() - cfg.dispatch_entropy_coef * entropy[policy_depart].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if verbose and update % 500 == 0:
                print(f"update {update:5d}  baseline={baseline:.3f}  n_decisions={int(policy_depart.sum().item())}")

    return policy


@torch.no_grad()
def inter_departure_variance(cfg: Config, net, device: torch.device, policy: DispatchPolicy) -> dict:
    """Run the trained policy for one simulated day (at `cfg.fleet_size`,
    the same size it was trained at) and report the variance of
    inter-departure gaps — the metric §9b.3 tracks vs. population size.

    Returns a decomposition, not just the pooled number: pooling gaps across
    every vehicle on every corridor conflates two different things — each
    vehicle's own irregularity (within-corridor variance) and corridors
    simply having different mean gaps (between-corridor variance). A pooled
    number rising with fleet size could mean either "each operator got more
    erratic" or "corridors just diverged in their average pace"; only the
    decomposition tells you which. Also reports the mean gap and the
    coefficient of variation (std/mean), since raw variance conflates
    genuine irregularity with gaps simply being larger or smaller in scale.
    """
    fleet_size = cfg.fleet_size
    gen = resolve_generator(cfg, device, seed_offset=1)
    fleet = init_fleet(net, cfg, cfg.n_days, fleet_size, device)
    queue = torch.zeros(cfg.n_days, n_demand_stops(net), device=device)
    t_hours = cfg.day_start_hour

    corridor_of_vehicle = fleet.corridor[0].clone()  # [fleet_size], constant over time
    last_departure_time = torch.full((cfg.n_days, fleet_size), float("nan"), device=device)
    gaps: list[float] = []
    gap_corridors: list[int] = []

    for _ in range(cfg.n_steps):
        fleet, queue, boarded, depart_now, policy_depart, logprob, entropy, reward = step_learned(
            fleet, queue, net, cfg, gen, policy, t_hours
        )
        # Record a gap (time since this vehicle's previous departure, of any
        # cause) only when the *current* departure was the policy's own
        # choice, so forced max-wait fallbacks don't get counted as emergent
        # policy behaviour, but they still correctly reset the clock.
        if depart_now.any():
            has_prev = ~torch.isnan(last_departure_time)
            record = policy_depart & has_prev
            if record.any():
                gaps.extend((t_hours - last_departure_time[record]).tolist())
                vehicle_idx = record.nonzero(as_tuple=True)[1]
                gap_corridors.extend(corridor_of_vehicle[vehicle_idx].tolist())
            last_departure_time = torch.where(depart_now, torch.full_like(last_departure_time, t_hours), last_departure_time)
        t_hours += cfg.step_minutes / 60.0

    if len(gaps) < 2:
        return {"pooled_variance": float("nan"), "n_gaps": len(gaps)}

    gaps_t = torch.tensor(gaps)
    corridors_t = torch.tensor(gap_corridors)
    pooled_variance = gaps_t.var(unbiased=True).item()
    mean_gap = gaps_t.mean().item()

    corridor_means, weighted_within_sum, weighted_within_n = [], 0.0, 0
    for c in corridors_t.unique().tolist():
        g = gaps_t[corridors_t == c]
        corridor_means.append(g.mean().item())
        if len(g) > 1:
            weighted_within_sum += g.var(unbiased=True).item() * len(g)
            weighted_within_n += len(g)
    within_corridor_variance = weighted_within_sum / weighted_within_n if weighted_within_n > 0 else float("nan")
    between_corridor_variance = torch.tensor(corridor_means).var(unbiased=False).item() if len(corridor_means) > 1 else float("nan")

    return {
        "pooled_variance": pooled_variance,
        "within_corridor_variance": within_corridor_variance,
        "between_corridor_variance": between_corridor_variance,
        "mean_gap_hours": mean_gap,
        "coefficient_of_variation": (pooled_variance ** 0.5) / mean_gap if mean_gap > 0 else float("nan"),
        "n_gaps": len(gaps),
        "n_corridors_observed": len(corridor_means),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fleet-sizes", type=int, nargs="+", default=[10, 20, 40, 80])
    parser.add_argument("--out", type=str, default="runs/dispatch_convergence.json")
    args = parser.parse_args()

    cfg = Config()
    if args.policy_steps is not None:
        cfg.policy_steps = args.policy_steps
    if args.device is not None:
        cfg.device = args.device
    if args.seed is not None:
        cfg.seed = args.seed

    device = cfg.resolved_device()
    gen_net = torch.Generator().manual_seed(cfg.seed)
    net = generate_network(cfg, gen_net)

    # A separate policy is trained from scratch at each fleet size, so each
    # reaches its own equilibrium under its own population of simultaneously
    # optimizing agents — evaluating one policy (trained at a single fleet
    # size) against other fleet sizes would test out-of-distribution
    # generalisation, not the §9b.3 population-size convergence question.
    results = {}
    for fleet_size in args.fleet_sizes:
        run_cfg = replace(cfg, fleet_size=fleet_size)
        print(f"Training dispatch policy for {run_cfg.policy_steps} steps on {device}, fleet_size={fleet_size}...")
        policy = train_policy(run_cfg, net, device, verbose=False)
        diagnostics = inter_departure_variance(run_cfg, net, device, policy)
        results[fleet_size] = diagnostics
        print(
            f"fleet_size={fleet_size:4d}  pooled_var={diagnostics['pooled_variance']:.4f}  "
            f"within={diagnostics.get('within_corridor_variance', float('nan')):.4f}  "
            f"between={diagnostics.get('between_corridor_variance', float('nan')):.4f}  "
            f"mean_gap={diagnostics.get('mean_gap_hours', float('nan')):.3f}h  "
            f"CV={diagnostics.get('coefficient_of_variation', float('nan')):.3f}  "
            f"n_gaps={diagnostics['n_gaps']}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"config": {k: v for k, v in vars(cfg).items()}, "inter_departure_variance": results}, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
