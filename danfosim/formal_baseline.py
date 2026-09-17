"""Formal fixed-route, fixed-headway baseline (§6): the comparison point for
the central informal-vs-formal experiment (§9a). Reuses the same corridor
route structure and the shared tensor primitives from `vehicle.py`, but:

    - departs a stop at a fixed headway, regardless of passenger load
    - fixed route only, no probabilistic mid-route deviation
    - hard capacity cap: excess demand queues for the next scheduled vehicle
      instead of overloading this one
"""

from __future__ import annotations

import torch

from .config import Config
from .demand import INBOUND, OUTBOUND
from .network import Network
from .vehicle import (
    EN_ROUTE,
    WAITING,
    VehicleState,
    leg_destination_j,
    next_local_j,
    board_waiting_vehicles,
    compute_density,
)


def init_fleet(net: Network, cfg: Config, batch_size: int, fleet_size: int, device: torch.device) -> VehicleState:
    """Round-robin corridor assignment (matching the informal fleet, for a fair
    equal-fleet-size comparison), with vehicles on the same corridor staggered
    across the headway cycle so the schedule isn't bunched at t=0."""
    idx = torch.arange(fleet_size, device=device)
    corridor_1d = idx % net.n_corridors
    rank = idx // net.n_corridors
    n_per_corridor = torch.bincount(corridor_1d, minlength=net.n_corridors).float()
    per_corridor_count = n_per_corridor[corridor_1d].clamp(min=1.0)
    initial_wait = (rank.float() * cfg.headway_minutes / per_corridor_count) % cfg.headway_minutes

    corridor = corridor_1d.long().unsqueeze(0).expand(batch_size, -1).contiguous()
    wait_time = initial_wait.unsqueeze(0).expand(batch_size, -1).contiguous().clone()
    zeros = torch.zeros(batch_size, fleet_size, device=device)
    return VehicleState(
        corridor=corridor,
        direction=torch.full((batch_size, fleet_size), INBOUND, dtype=torch.long, device=device),
        cur_j=torch.zeros(batch_size, fleet_size, dtype=torch.long, device=device),
        state=torch.full((batch_size, fleet_size), WAITING, dtype=torch.long, device=device),
        onboard=zeros.clone(),
        wait_time=wait_time,
        remaining_travel_time=zeros.clone(),
        trips_completed=zeros.clone(),
        travel_time_accum=zeros.clone(),
        load_at_departure_sum=zeros.clone(),
        n_departures=zeros.clone(),
    )


def step_formal(
    state: VehicleState,
    queue: torch.Tensor,
    net: Network,
    cfg: Config,
    gen: torch.Generator,
) -> tuple[VehicleState, torch.Tensor, torch.Tensor]:
    """Advance the formal fleet and passenger queues by one step.
    Returns (new_state, new_queue, boarded_count_per_batch)."""
    J = net.n_junctions

    waiting_mask = state.state == WAITING
    enroute_mask = state.state == EN_ROUTE

    density = compute_density(state, net, enroute_mask)

    queue, boarded_count = board_waiting_vehicles(state, queue, net, waiting_mask, cfg.hard_capacity)

    state.wait_time = torch.where(waiting_mask, state.wait_time + cfg.step_minutes, state.wait_time)

    # A fixed-route bus boards at every junction along its route, not just the
    # terminus: `headway_minutes` governs departure from the terminus (where a
    # fresh leg begins after reversing direction), while `dwell_minutes` is a
    # brief, load-independent boarding stop at each intermediate junction.
    at_terminus = ((state.direction == INBOUND) & (state.cur_j == 0)) | (
        (state.direction == OUTBOUND) & (state.cur_j == J - 1)
    )
    required_wait = torch.where(
        at_terminus, torch.full_like(state.wait_time, cfg.headway_minutes), torch.full_like(state.wait_time, cfg.dwell_minutes)
    )
    depart_now = waiting_mask & (state.wait_time >= required_wait)

    dep_edge = net.edge_for_leg(state.corridor, state.cur_j, state.direction)
    dep_density = density.gather(1, dep_edge)
    dep_travel_time = net.travel_time(dep_edge, dep_density)

    load_frac = state.onboard / cfg.hard_capacity
    state.remaining_travel_time = torch.where(depart_now, dep_travel_time, state.remaining_travel_time)
    # utilisation/departure-count metrics reflect scheduled (terminus) departures,
    # since intermediate dwell stops aren't a scheduling decision
    terminus_departure = depart_now & at_terminus
    state.load_at_departure_sum = torch.where(terminus_departure, state.load_at_departure_sum + load_frac, state.load_at_departure_sum)
    state.n_departures = torch.where(terminus_departure, state.n_departures + 1, state.n_departures)
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
    mid_route_arrival = arrived_mask & ~trip_complete  # dwell to board, not a full trip end

    state.trips_completed = torch.where(trip_complete, state.trips_completed + 1, state.trips_completed)
    state.onboard = torch.where(trip_complete, torch.zeros_like(state.onboard), state.onboard)

    state.cur_j = torch.where(trip_complete, destination_j, state.cur_j)
    state.direction = torch.where(trip_complete, 1 - state.direction, state.direction)
    state.cur_j = torch.where(mid_route_arrival, nxt_j, state.cur_j)

    state.state = torch.where(trip_complete | mid_route_arrival, torch.full_like(state.state, WAITING), state.state)
    state.wait_time = torch.where(trip_complete | mid_route_arrival, torch.zeros_like(state.wait_time), state.wait_time)

    return state, queue, boarded_count
