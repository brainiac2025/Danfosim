"""§8: train a learned dispatch policy for the informal fleet, replacing the
fixed threshold-departure rule, and measure whether its emergent departure
pattern becomes more regular (lower inter-departure variance) as more agents
optimize simultaneously (§9b.3: inter-departure variance vs. population size).
"""

from __future__ import annotations

import argparse
import json
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
def inter_departure_variance(cfg: Config, net, device: torch.device, policy: DispatchPolicy, fleet_size: int) -> float:
    """Run the trained policy for one simulated day at the given fleet size
    and report the variance of inter-departure gaps, pooled across all
    corridors and batch days — the metric §9b.3 tracks vs. population size."""
    cfg_run = Config(**{**vars(cfg), "fleet_size": fleet_size})
    gen = resolve_generator(cfg_run, device, seed_offset=1)
    fleet = init_fleet(net, cfg_run, cfg_run.n_days, fleet_size, device)
    queue = torch.zeros(cfg_run.n_days, n_demand_stops(net), device=device)
    t_hours = cfg_run.day_start_hour

    last_departure_time = torch.full((cfg_run.n_days, fleet_size), float("nan"), device=device)
    gaps: list[float] = []

    for _ in range(cfg_run.n_steps):
        fleet, queue, boarded, depart_now, policy_depart, logprob, entropy, reward = step_learned(
            fleet, queue, net, cfg_run, gen, policy, t_hours
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
            last_departure_time = torch.where(depart_now, torch.full_like(last_departure_time, t_hours), last_departure_time)
        t_hours += cfg_run.step_minutes / 60.0

    if len(gaps) < 2:
        return float("nan")
    gaps_t = torch.tensor(gaps)
    return gaps_t.var(unbiased=True).item()


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

    print(f"Training dispatch policy for {cfg.policy_steps} steps on {device}...")
    policy = train_policy(cfg, net, device)

    results = {}
    for fleet_size in args.fleet_sizes:
        variance = inter_departure_variance(cfg, net, device, policy, fleet_size)
        results[fleet_size] = variance
        print(f"fleet_size={fleet_size:4d}  inter-departure variance (hours^2) = {variance:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"config": {k: v for k, v in vars(cfg).items()}, "inter_departure_variance": results}, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
