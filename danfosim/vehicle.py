"""Informal vehicle agents (danfo/keke): threshold departure + route deviation.

State machine (batched over [n_days, fleet_size] as tensor ops — no Python
loop over individual vehicles or passengers):

    WAITING_AT_STOP (accumulating passengers)
        -> DEPART (load threshold met, or max_wait exceeded)
    EN_ROUTE (subject to congestion)
        -> arrival at next junction:
            - trip complete (reached the leg's destination) -> unload,
              reverse direction, WAITING_AT_STOP at the new position
            - mid-route, with probability `deviation_prob` -> DROP: end the
              trip early at the current junction (informal short-turn toward
              local demand) -> WAITING_AT_STOP at the same position
            - otherwise -> a rolling curbside pickup at the junction just
              reached (no dwell-time cost, unlike a formal bus's scheduled
              stop), then continue onto the next edge, still EN_ROUTE

A "drop" is counted as a completed trip. This is a simplification: a real
dropped passenger may need to find a second vehicle to finish their journey,
which this model does not represent (see docs/CALIBRATION.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import Config
from .demand import INBOUND, OUTBOUND, demand_stop_id
from .network import Network

WAITING = 0
EN_ROUTE = 1


@dataclass
class VehicleState:
    corridor: torch.Tensor  # [B, F] long
    direction: torch.Tensor  # [B, F] long, INBOUND or OUTBOUND
    cur_j: torch.Tensor  # [B, F] long, local junction index of current leg's start
    state: torch.Tensor  # [B, F] long, WAITING or EN_ROUTE
    onboard: torch.Tensor  # [B, F] float, passengers currently aboard
    wait_time: torch.Tensor  # [B, F] float, minutes waited at the current stop
    remaining_travel_time: torch.Tensor  # [B, F] float, minutes left on current edge
    trips_completed: torch.Tensor  # [B, F] float, cumulative counter
    travel_time_accum: torch.Tensor  # [B, F] float, cumulative minutes spent en route
    load_at_departure_sum: torch.Tensor  # [B, F] float, sum of onboard/capacity at each departure
    n_departures: torch.Tensor  # [B, F] float, count of departures (for utilisation mean)

    @property
    def batch_size(self) -> int:
        return self.corridor.shape[0]

    @property
    def fleet_size(self) -> int:
        return self.corridor.shape[1]

    def to(self, device: torch.device) -> "VehicleState":
        return VehicleState(**{f: getattr(self, f).to(device) for f in self.__dataclass_fields__})


def init_fleet(net: Network, cfg: Config, batch_size: int, fleet_size: int, device: torch.device) -> VehicleState:
    """Assign vehicles round-robin across corridors, all starting WAITING at
    the outermost (periphery) junction of their corridor, inbound."""
    idx = torch.arange(fleet_size, device=device)
    corridor = (idx % net.n_corridors).long().unsqueeze(0).expand(batch_size, -1).contiguous()
    zeros = torch.zeros(batch_size, fleet_size, device=device)
    return VehicleState(
        corridor=corridor,
        direction=torch.full((batch_size, fleet_size), INBOUND, dtype=torch.long, device=device),
        cur_j=torch.zeros(batch_size, fleet_size, dtype=torch.long, device=device),
        state=torch.full((batch_size, fleet_size), WAITING, dtype=torch.long, device=device),
        onboard=zeros.clone(),
        wait_time=zeros.clone(),
        remaining_travel_time=zeros.clone(),
        trips_completed=zeros.clone(),
        travel_time_accum=zeros.clone(),
        load_at_departure_sum=zeros.clone(),
        n_departures=zeros.clone(),
    )


def next_local_j(local_j: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    return torch.where(direction == INBOUND, local_j + 1, local_j - 1)


def leg_destination_j(direction: torch.Tensor, n_junctions: int) -> torch.Tensor:
    return torch.where(direction == INBOUND, torch.full_like(direction, n_junctions - 1), torch.zeros_like(direction))


def compute_density(state: VehicleState, net: Network, enroute_mask: torch.Tensor, cfg: Config, t_hours: float) -> torch.Tensor:
    """Vehicle count per edge, per batch row, plus a time-varying background
    (non-transit) traffic term (see `Network.ambient_density`). Shape [B, E]."""
    B = state.batch_size
    edge_idx = net.edge_for_leg(state.corridor, state.cur_j, state.direction)
    ones = torch.where(enroute_mask, torch.ones_like(edge_idx, dtype=torch.float), torch.zeros(1, device=edge_idx.device))
    density = torch.zeros(B, net.n_edges, device=edge_idx.device)
    density.scatter_add_(1, edge_idx, ones)
    return density + net.ambient_density(cfg, t_hours).unsqueeze(0)


def board_at_positions(
    state: VehicleState,
    queue: torch.Tensor,
    net: Network,
    mask: torch.Tensor,
    position_j: torch.Tensor,
    max_capacity: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move passengers from stop queues onto vehicles selected by `mask`,
    boarding at junction `position_j` (which may differ from `state.cur_j`,
    e.g. a rolling pickup at the junction a vehicle is just passing through).
    Handles multiple vehicles sharing a stop by allocating queue proportionally
    to requested room, so no passenger is double-counted. Returns
    (updated_queue, boarded_count_per_batch); `queue` is not mutated in place."""
    B = state.batch_size
    S = queue.shape[1]
    stop_id = demand_stop_id(state.corridor, position_j, net.n_junctions)

    room = (max_capacity - state.onboard).clamp(min=0.0)
    room = torch.where(mask, room, torch.zeros_like(room))

    requested_per_stop = torch.zeros(B, S, device=queue.device)
    requested_per_stop.scatter_add_(1, stop_id, room)

    frac = torch.where(
        requested_per_stop > 0,
        (queue / requested_per_stop.clamp(min=1e-6)).clamp(max=1.0),
        torch.zeros_like(queue),
    )
    frac_per_vehicle = frac.gather(1, stop_id)
    boarded = room * frac_per_vehicle

    state.onboard = state.onboard + boarded

    boarded_per_stop = torch.zeros(B, S, device=queue.device)
    boarded_per_stop.scatter_add_(1, stop_id, boarded)
    return (queue - boarded_per_stop).clamp(min=0.0), boarded_per_stop.sum(dim=1)


def board_waiting_vehicles(
    state: VehicleState,
    queue: torch.Tensor,
    net: Network,
    waiting_mask: torch.Tensor,
    max_capacity: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience wrapper: board vehicles at their current stop, `state.cur_j`."""
    return board_at_positions(state, queue, net, waiting_mask, state.cur_j, max_capacity)


def board_and_prepare(
    state: VehicleState,
    queue: torch.Tensor,
    net: Network,
    cfg: Config,
    enroute_mask: torch.Tensor,
    waiting_mask: torch.Tensor,
    t_hours: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared first half of an informal-regime step: compute edge density,
    board waiting vehicles at their current stop, and accrue wait time.
    Returns (new_queue, boarded_count_per_batch, density)."""
    density = compute_density(state, net, enroute_mask, cfg, t_hours)
    max_capacity = cfg.nominal_capacity * cfg.soft_capacity_overload
    queue, boarded_count = board_waiting_vehicles(state, queue, net, waiting_mask, max_capacity)
    state.wait_time = torch.where(waiting_mask, state.wait_time + cfg.step_minutes, state.wait_time)
    return queue, boarded_count, density


def apply_departure_and_travel(
    state: VehicleState,
    queue: torch.Tensor,
    net: Network,
    cfg: Config,
    gen: torch.Generator,
    depart_now: torch.Tensor,
    density: torch.Tensor,
    load_frac_at_decision: torch.Tensor,
    enroute_mask: torch.Tensor,
    boarded_count: torch.Tensor,
) -> tuple[VehicleState, torch.Tensor, torch.Tensor]:
    """Shared second half of an informal-regime step, given a `depart_now`
    mask decided by the caller's departure policy (fixed threshold rule, or
    a learned policy): apply the departure, advance en-route vehicles,
    resolve arrivals (trip completion / probabilistic deviation / rolling
    curbside pickup). Returns (new_state, new_queue, boarded_count_per_batch)."""
    J = net.n_junctions

    dep_edge = net.edge_for_leg(state.corridor, state.cur_j, state.direction)
    dep_density = density.gather(1, dep_edge)
    dep_travel_time = net.travel_time(dep_edge, dep_density)

    state.remaining_travel_time = torch.where(depart_now, dep_travel_time, state.remaining_travel_time)
    state.load_at_departure_sum = torch.where(
        depart_now, state.load_at_departure_sum + load_frac_at_decision, state.load_at_departure_sum
    )
    state.n_departures = torch.where(depart_now, state.n_departures + 1, state.n_departures)
    state.state = torch.where(depart_now, torch.full_like(state.state, EN_ROUTE), state.state)
    state.wait_time = torch.where(depart_now, torch.zeros_like(state.wait_time), state.wait_time)

    state.travel_time_accum = torch.where(enroute_mask, state.travel_time_accum + cfg.step_minutes, state.travel_time_accum)
    state.remaining_travel_time = torch.where(
        enroute_mask, (state.remaining_travel_time - cfg.step_minutes).clamp(min=0.0), state.remaining_travel_time
    )
    arrived_mask = enroute_mask & (state.remaining_travel_time <= 0)

    nxt_j = next_local_j(state.cur_j, state.direction)
    destination_j = leg_destination_j(state.direction, J)
    trip_complete = arrived_mask & (nxt_j == destination_j)
    mid_trip_arrival = arrived_mask & ~trip_complete

    rand_dev = torch.rand(state.corridor.shape, generator=gen, device=state.corridor.device)
    deviate = mid_trip_arrival & (rand_dev < cfg.deviation_prob)
    continue_enroute = mid_trip_arrival & ~deviate

    # Rolling curbside pickup: a danfo passing an intermediate junction without
    # deviating still picks up any waiting passengers there in passing (no
    # dwell-time cost), unlike a formal bus's scheduled stop. This keeps
    # intermediate-junction demand from starving under the informal regime.
    max_capacity = cfg.nominal_capacity * cfg.soft_capacity_overload
    queue, rolling_boarded = board_at_positions(state, queue, net, continue_enroute, nxt_j, max_capacity)
    boarded_count = boarded_count + rolling_boarded

    state.trips_completed = torch.where(trip_complete | deviate, state.trips_completed + 1, state.trips_completed)
    state.onboard = torch.where(trip_complete | deviate, torch.zeros_like(state.onboard), state.onboard)

    state.cur_j = torch.where(trip_complete, destination_j, state.cur_j)
    state.direction = torch.where(trip_complete, 1 - state.direction, state.direction)
    state.cur_j = torch.where(deviate, nxt_j, state.cur_j)
    state.cur_j = torch.where(continue_enroute, nxt_j, state.cur_j)

    state.state = torch.where(trip_complete | deviate, torch.full_like(state.state, WAITING), state.state)
    state.wait_time = torch.where(trip_complete | deviate, torch.zeros_like(state.wait_time), state.wait_time)

    cont_edge = net.edge_for_leg(state.corridor, state.cur_j, state.direction)
    cont_density = density.gather(1, cont_edge)
    state.remaining_travel_time = torch.where(
        continue_enroute, net.travel_time(cont_edge, cont_density), state.remaining_travel_time
    )

    return state, queue, boarded_count


def step_informal(
    state: VehicleState,
    queue: torch.Tensor,
    net: Network,
    cfg: Config,
    gen: torch.Generator,
    t_hours: float,
) -> tuple[VehicleState, torch.Tensor, torch.Tensor]:
    """Advance the informal fleet and passenger queues by one step, using the
    fixed load-threshold departure rule (§5). Returns (new_state, new_queue,
    boarded_count_per_batch)."""
    waiting_mask = state.state == WAITING
    enroute_mask = state.state == EN_ROUTE

    queue, boarded_count, density = board_and_prepare(state, queue, net, cfg, enroute_mask, waiting_mask, t_hours)

    load_frac = state.onboard / cfg.nominal_capacity
    depart_now = waiting_mask & (
        (load_frac >= cfg.departure_threshold) | (state.wait_time >= cfg.max_wait_before_departure_anyway)
    )

    return apply_departure_and_travel(state, queue, net, cfg, gen, depart_now, density, load_frac, enroute_mask, boarded_count)
