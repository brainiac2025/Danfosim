"""Batched multi-agent simulation loop (§6). Runs `cfg.n_days` independent
simulated days simultaneously as tensor state (same network, independent
demand realisations per batch row), advancing all vehicles and passenger
queues in lockstep. All per-step work is `[B, *]`-shaped tensor ops — there
is no Python loop over individual vehicles or passengers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from . import formal_baseline, vehicle
from .config import Config
from .demand import n_demand_stops, sample_arrivals
from .network import Network


@dataclass
class SimState:
    net: Network
    cfg: Config
    device: torch.device
    regime: str  # "informal" or "formal"
    fleet: vehicle.VehicleState
    queue: torch.Tensor  # [B, S]
    t_hours: float
    gen: torch.Generator


def resolve_generator(cfg: Config, device: torch.device, seed_offset: int = 0) -> torch.Generator:
    gen = torch.Generator(device=device if device.type != "cpu" else "cpu")
    gen.manual_seed(cfg.seed + seed_offset)
    return gen


def init_sim(cfg: Config, net: Network, regime: str, device: torch.device, seed_offset: int = 0) -> SimState:
    if regime not in ("informal", "formal"):
        raise ValueError(f"unknown regime {regime!r}")
    gen = resolve_generator(cfg, device, seed_offset)
    net_dev = net.to(device)
    if regime == "informal":
        fleet = vehicle.init_fleet(net_dev, cfg, cfg.n_days, cfg.fleet_size, device)
    else:
        fleet = formal_baseline.init_fleet(net_dev, cfg, cfg.n_days, cfg.fleet_size, device)
    queue = torch.zeros(cfg.n_days, n_demand_stops(net_dev), device=device)
    return SimState(
        net=net_dev, cfg=cfg, device=device, regime=regime, fleet=fleet, queue=queue,
        t_hours=cfg.day_start_hour, gen=gen,
    )


def step(sim_state: SimState) -> tuple[SimState, torch.Tensor]:
    """Advance the simulation by one step. Returns (new_state, boarded_per_day),
    where `boarded_per_day` is the count of passengers who boarded this step,
    shape [B]."""
    cfg, net = sim_state.cfg, sim_state.net
    arrivals = sample_arrivals(net, cfg, sim_state.t_hours, cfg.n_days, sim_state.gen, sim_state.device)
    sim_state.queue = sim_state.queue + arrivals

    step_fn = vehicle.step_informal if sim_state.regime == "informal" else formal_baseline.step_formal
    fleet, queue, boarded = step_fn(sim_state.fleet, sim_state.queue, net, cfg, sim_state.gen)

    sim_state.fleet = fleet
    sim_state.queue = queue
    sim_state.t_hours = sim_state.t_hours + cfg.step_minutes / 60.0
    return sim_state, boarded


@dataclass
class DayMetrics:
    """Per-step series (each entry is a [B] tensor), so callers can bucket by
    time-of-day (e.g. peak vs. off-peak) after the run."""

    time_hours: list = field(default_factory=list)
    boarded: list = field(default_factory=list)
    queue_person_minutes: list = field(default_factory=list)  # pre-boarding queue * step_minutes
    trips_completed: list = field(default_factory=list)  # diff of cumulative counter
    travel_minutes: list = field(default_factory=list)  # diff of cumulative in-vehicle minutes
    departures: list = field(default_factory=list)  # diff of cumulative departure counter
    load_at_departure_sum: list = field(default_factory=list)  # diff of cumulative load-at-departure sum


def run_day(cfg: Config, net: Network, regime: str, device: torch.device, seed_offset: int = 0) -> tuple[SimState, DayMetrics]:
    sim_state = init_sim(cfg, net, regime, device, seed_offset)
    metrics = DayMetrics()

    prev_trips = sim_state.fleet.trips_completed.sum(dim=1).clone()
    prev_travel = sim_state.fleet.travel_time_accum.sum(dim=1).clone()
    prev_dep = sim_state.fleet.n_departures.sum(dim=1).clone()
    prev_load = sim_state.fleet.load_at_departure_sum.sum(dim=1).clone()

    for _ in range(cfg.n_steps):
        t_hours = sim_state.t_hours
        sim_state, boarded = step(sim_state)
        # Person-minutes waited this step is approximated as (queue remaining
        # after this step's arrivals and boarding) * step_minutes — a standard
        # discrete-time Little's-law approximation, not an exact per-passenger
        # wait-time trace (see docs/CALIBRATION.md).

        cur_trips = sim_state.fleet.trips_completed.sum(dim=1)
        cur_travel = sim_state.fleet.travel_time_accum.sum(dim=1)
        cur_dep = sim_state.fleet.n_departures.sum(dim=1)
        cur_load = sim_state.fleet.load_at_departure_sum.sum(dim=1)

        metrics.time_hours.append(t_hours)
        metrics.boarded.append(boarded.detach().clone())
        metrics.queue_person_minutes.append((sim_state.queue.sum(dim=1) * cfg.step_minutes).detach().clone())
        metrics.trips_completed.append((cur_trips - prev_trips).detach().clone())
        metrics.travel_minutes.append((cur_travel - prev_travel).detach().clone())
        metrics.departures.append((cur_dep - prev_dep).detach().clone())
        metrics.load_at_departure_sum.append((cur_load - prev_load).detach().clone())

        prev_trips, prev_travel, prev_dep, prev_load = cur_trips, cur_travel, cur_dep, cur_load

    return sim_state, metrics


@dataclass
class Summary:
    mean_wait_time_minutes: float
    mean_travel_time_minutes: float
    mean_utilisation: float
    throughput_trips: float


def summarize(metrics: DayMetrics, cfg: Config, t_start: float | None = None, t_end: float | None = None) -> Summary:
    """Summarise a window of a completed run (all days in the batch pooled).

    Mean wait time uses the discrete Little's-law identity: total person-minutes
    spent in queue during the window, divided by the number of passengers who
    boarded during that same window.
    """
    idx = [
        i for i, t in enumerate(metrics.time_hours)
        if (t_start is None or t >= t_start) and (t_end is None or t < t_end)
    ]
    if not idx:
        return Summary(float("nan"), float("nan"), float("nan"), 0.0)

    total_boarded = sum(metrics.boarded[i].sum().item() for i in idx)
    total_queue_person_minutes = sum(metrics.queue_person_minutes[i].sum().item() for i in idx)
    total_trips = sum(metrics.trips_completed[i].sum().item() for i in idx)
    total_travel_minutes = sum(metrics.travel_minutes[i].sum().item() for i in idx)
    total_departures = sum(metrics.departures[i].sum().item() for i in idx)
    total_load_sum = sum(metrics.load_at_departure_sum[i].sum().item() for i in idx)

    mean_wait_time = total_queue_person_minutes / total_boarded if total_boarded > 0 else float("nan")
    mean_travel_time = total_travel_minutes / total_trips if total_trips > 0 else float("nan")
    mean_utilisation = total_load_sum / total_departures if total_departures > 0 else float("nan")

    return Summary(
        mean_wait_time_minutes=mean_wait_time,
        mean_travel_time_minutes=mean_travel_time,
        mean_utilisation=mean_utilisation,
        throughput_trips=total_trips,
    )
