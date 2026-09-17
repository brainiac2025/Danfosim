"""Learned dispatch policy (§8): replaces the informal fleet's fixed
load-threshold departure rule with a small policy trained to decide, at each
step a vehicle is WAITING, whether to depart now or wait longer — framed as
a per-decision contextual bandit (as the architecture specifies), not a
multi-step RL problem: the policy is rewarded immediately, at the moment it
chooses to depart, by a revenue-like proxy (occupancy carried, minus a
per-minute waiting cost), and REINFORCE with a moving-average baseline
shapes the departure probability accordingly.

This directly targets the risk flagged in the architecture (§12): a reward
that only counted trip *count* would collapse to "always depart immediately"
(near-empty departures still count as trips). Penalising wait time while
rewarding occupancy creates a genuine tension — the same
wait-vs-fill-up-more tradeoff a real revenue-driven driver faces — so the
question of whether optimizing it converges toward regular, schedule-like
departure intervals (§9b.3) is a real empirical one, not assumed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import Config
from .demand import demand_stop_id
from .network import Network
from .vehicle import EN_ROUTE, WAITING, VehicleState, apply_departure_and_travel, board_and_prepare

N_FEATURES = 4


class DispatchPolicy(nn.Module):
    """Per-vehicle features -> logit of P(depart now)."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def make_features(state: VehicleState, queue: torch.Tensor, net: Network, cfg: Config, t_hours: float) -> torch.Tensor:
    stop_id = demand_stop_id(state.corridor, state.cur_j, net.n_junctions)
    local_queue = queue.gather(1, stop_id)
    load_frac = state.onboard / cfg.nominal_capacity
    wait_frac = state.wait_time / max(cfg.max_wait_before_departure_anyway, 1e-6)
    tod = torch.full_like(load_frac, (t_hours % 24.0) / 24.0)
    queue_frac = (local_queue / cfg.nominal_capacity).clamp(max=3.0)
    return torch.stack([load_frac, wait_frac, tod, queue_frac], dim=-1)


def step_learned(
    state: VehicleState,
    queue: torch.Tensor,
    net: Network,
    cfg: Config,
    gen: torch.Generator,
    policy: DispatchPolicy,
    t_hours: float,
) -> tuple[VehicleState, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One simulation step with departure decisions drawn from `policy`.

    Returns (state, queue, boarded_count, depart_now, policy_depart, logprob,
    entropy, reward). `depart_now` marks every vehicle that departed this
    step, by any cause; `policy_depart` is the subset that departed *by the
    policy's own choice* (excluding the max-wait safety fallback, which
    always applies so a trip can never stall forever) — `logprob`, `entropy`,
    and `reward` are defined for every vehicle but only meaningful, and only
    used for training, at positions where `policy_depart` is True.
    """
    waiting_mask = state.state == WAITING
    enroute_mask = state.state == EN_ROUTE

    queue, boarded_count, density = board_and_prepare(state, queue, net, cfg, enroute_mask, waiting_mask, t_hours)

    features = make_features(state, queue, net, cfg, t_hours)
    logits = policy(features)
    probs = torch.sigmoid(logits)
    sample = torch.bernoulli(probs.detach(), generator=gen)

    logprob = sample * torch.log(probs.clamp(min=1e-6)) + (1 - sample) * torch.log((1 - probs).clamp(min=1e-6))
    entropy = -(
        probs * torch.log(probs.clamp(min=1e-6)) + (1 - probs) * torch.log((1 - probs).clamp(min=1e-6))
    )

    policy_depart = (sample > 0.5) & waiting_mask
    forced_depart = waiting_mask & (state.wait_time >= cfg.max_wait_before_departure_anyway)
    depart_now = policy_depart | forced_depart

    load_frac = state.onboard / cfg.nominal_capacity
    reward = load_frac - cfg.dispatch_time_penalty * state.wait_time

    state, queue, boarded_count = apply_departure_and_travel(
        state, queue, net, cfg, gen, depart_now, density, load_frac, enroute_mask, boarded_count
    )

    return state, queue, boarded_count, depart_now, policy_depart, logprob, entropy, reward
