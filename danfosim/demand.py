"""Passenger arrival generation: time-varying Poisson process per demand stop.

Demand stops: each corridor contributes `n_junctions` demand stops —
`n_junctions - 1` inbound-boarding junctions (passengers waiting to travel
toward the CBD) plus one CBD outbound-boarding "stop" specific to that
corridor (passengers waiting to travel back out that corridor in the
evening). This is the single most robust, well-documented fact about Lagos
commute patterns worth calibrating confidently: a bimodal peak (morning
inbound-to-CBD, evening outbound).
"""

from __future__ import annotations

import math

import torch

from .config import Config
from .network import Network

INBOUND = 0
OUTBOUND = 1


def n_demand_stops(net: Network) -> int:
    return net.n_corridors * net.n_junctions


def demand_stop_id(corridor: torch.Tensor, local_j: torch.Tensor, n_junctions: int) -> torch.Tensor:
    """Maps (corridor, local_j) -> a flat demand-stop id. local_j == n_junctions-1
    means the CBD outbound queue specific to that corridor."""
    return corridor * n_junctions + local_j


def _stop_meta(net: Network) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per demand-stop-id: (corridor, local_j, is_outbound)."""
    K, J = net.n_corridors, net.n_junctions
    idx = torch.arange(K * J)
    corridor = idx // J
    local_j = idx % J
    is_outbound = local_j == (J - 1)
    return corridor, local_j, is_outbound


def _gaussian(t: float, peak: float, width: float) -> float:
    return math.exp(-0.5 * ((t - peak) / width) ** 2)


def arrival_rate(net: Network, cfg: Config, t_hours: float) -> torch.Tensor:
    """Per-minute Poisson arrival rate at every demand stop at time `t_hours`.
    Shape [S] (S = n_demand_stops), CPU float tensor."""
    J = net.n_junctions
    _, local_j, is_outbound = _stop_meta(net)
    hops_from_cbd = (J - 1 - local_j).clamp(min=0).float()
    # Outer (suburban) stops carry modestly more inbound demand than
    # CBD-adjacent ones; outbound (CBD) queues are treated uniformly per corridor.
    distance_factor = torch.where(
        is_outbound,
        torch.ones_like(hops_from_cbd),
        0.5 + 0.5 * hops_from_cbd / max(J - 1, 1),
    )
    morning = _gaussian(t_hours, cfg.morning_peak_hour, cfg.peak_width_hours)
    evening = _gaussian(t_hours, cfg.evening_peak_hour, cfg.peak_width_hours)
    peak_component = torch.where(
        is_outbound,
        1.0 + (cfg.peak_multiplier - 1.0) * evening,
        1.0 + (cfg.peak_multiplier - 1.0) * morning,
    )
    return cfg.base_arrival_rate * distance_factor * peak_component


def sample_arrivals(
    net: Network,
    cfg: Config,
    t_hours: float,
    batch_size: int,
    gen: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """New passengers arriving at each demand stop this step. Shape [B, S]."""
    rate_per_min = arrival_rate(net, cfg, t_hours).to(device)
    lam = (rate_per_min * cfg.step_minutes).clamp(min=0.0)
    lam = lam.unsqueeze(0).expand(batch_size, -1).contiguous()
    return torch.poisson(lam, generator=gen)
