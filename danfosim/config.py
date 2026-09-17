from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Config:
    # network (§3)
    n_corridors: int = 6
    n_junctions_per_corridor: int = 8
    bottleneck_edges_frac: float = 0.15
    # background (non-transit) road traffic contributing to edge congestion,
    # since a Lagos bridge's chronic congestion comes overwhelmingly from
    # general traffic, not the small simulated transit fleet's own density
    ambient_traffic_base: float = 1.5  # vehicle-equivalent density, off-peak, non-bottleneck edge
    ambient_traffic_bottleneck_multiplier: float = 3.0  # bottleneck edges carry more background traffic
    ambient_traffic_peak_multiplier: float = 2.0  # background traffic also rises at rush hour

    # demand (§4)
    base_arrival_rate: float = 0.35  # passengers/minute at an average off-peak stop
    peak_multiplier: float = 3.0
    morning_peak_hour: float = 7.5
    evening_peak_hour: float = 17.5
    peak_width_hours: float = 1.2  # gaussian std-dev of each peak bump

    # informal vehicle (§5)
    nominal_capacity: float = 14.0  # typical danfo minibus seating
    departure_threshold: float = 0.75
    max_wait_before_departure_anyway: float = 8.0  # minutes
    deviation_prob: float = 0.15
    soft_capacity_overload: float = 1.2

    # formal vehicle (§6)
    headway_minutes: float = 15.0
    hard_capacity: float = 30.0
    dwell_minutes: float = 0.5  # brief boarding stop at each intermediate junction

    # sim (§7)
    n_days: int = 200  # parallel batch
    day_start_hour: float = 5.0
    sim_hours: float = 16.0
    step_minutes: float = 1.0
    fleet_size: int = 40

    # dispatch policy (§8)
    policy_lr: float = 1e-3
    policy_steps: int = 5000
    policy_hidden: int = 16
    dispatch_time_penalty: float = 0.05  # per-minute-waited cost in the reward
    dispatch_entropy_coef: float = 0.01

    device: str = "cuda"
    seed: int = 0

    @property
    def n_steps(self) -> int:
        return int(round(self.sim_hours * 60.0 / self.step_minutes))

    def resolved_device(self) -> torch.device:
        if self.device == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(self.device)
