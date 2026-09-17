import torch

from danfosim.config import Config
from danfosim.demand import n_demand_stops, sample_arrivals
from danfosim.network import generate_network
from danfosim.vehicle import EN_ROUTE, WAITING, init_fleet, step_informal
from danfosim import formal_baseline


def _setup(**overrides):
    cfg = Config(**overrides)
    gen_net = torch.Generator().manual_seed(cfg.seed)
    net = generate_network(cfg, gen_net)
    device = torch.device("cpu")
    gen = torch.Generator(device=device).manual_seed(cfg.seed + 1)
    return cfg, net, device, gen


def test_init_fleet_starts_waiting_inbound_at_periphery():
    cfg, net, device, gen = _setup(fleet_size=12)
    state = init_fleet(net, cfg, batch_size=3, fleet_size=12, device=device)
    assert (state.state == WAITING).all()
    assert (state.cur_j == 0).all()
    assert (state.onboard == 0).all()


def test_vehicle_departs_once_threshold_met_with_no_max_wait_cap():
    cfg, net, device, gen = _setup(fleet_size=1, n_days=1, departure_threshold=0.5,
                                    max_wait_before_departure_anyway=1e9, base_arrival_rate=50.0)
    state = init_fleet(net, cfg, batch_size=1, fleet_size=1, device=device)
    queue = torch.zeros(1, n_demand_stops(net), device=device)
    # force a big queue at the vehicle's stop so it fills past threshold immediately
    queue[0, 0] = 100.0
    state, queue, boarded = step_informal(state, queue, net, cfg, gen)
    assert state.state[0, 0].item() == EN_ROUTE
    assert state.onboard[0, 0].item() > 0.0  # boarded passengers stay onboard once en route
    assert state.onboard[0, 0].item() == boarded[0].item()


def test_vehicle_departs_on_max_wait_even_with_no_passengers():
    cfg, net, device, gen = _setup(fleet_size=1, n_days=1, departure_threshold=0.99,
                                    max_wait_before_departure_anyway=2.0, base_arrival_rate=0.0)
    state = init_fleet(net, cfg, batch_size=1, fleet_size=1, device=device)
    queue = torch.zeros(1, n_demand_stops(net), device=device)
    for _ in range(2):
        state, queue, boarded = step_informal(state, queue, net, cfg, gen)
    assert state.state[0, 0].item() == EN_ROUTE


def test_boarding_never_creates_or_destroys_passengers():
    cfg, net, device, gen = _setup(fleet_size=20, n_days=4, base_arrival_rate=5.0)
    state = init_fleet(net, cfg, batch_size=4, fleet_size=20, device=device)
    queue = torch.zeros(4, n_demand_stops(net), device=device)
    total_arrived = torch.zeros(4, device=device)
    for step in range(50):
        t_hours = cfg.day_start_hour + step * cfg.step_minutes / 60.0
        arrivals = sample_arrivals(net, cfg, t_hours, batch_size=4, gen=gen, device=device)
        queue = queue + arrivals
        total_arrived += arrivals.sum(dim=1)
        state, queue, boarded = step_informal(state, queue, net, cfg, gen)
    still_onboard = state.onboard.sum(dim=1)
    still_queued = queue.sum(dim=1)
    completed_or_dropped = state.trips_completed.sum(dim=1)  # each trip carried >=0 passengers, not a count check
    # conservation: everyone who arrived is either still queued, still onboard, or was
    # aboard a vehicle that finished a trip (we don't track per-trip payload directly,
    # so check the weaker but still meaningful invariant: nothing goes negative and
    # queued+onboard never exceeds total arrived).
    assert (still_onboard >= 0).all()
    assert (still_queued >= 0).all()
    assert (still_queued + still_onboard <= total_arrived + 1e-4).all()


def test_soft_overload_allows_exceeding_nominal_capacity():
    cfg, net, device, gen = _setup(fleet_size=1, n_days=1, departure_threshold=0.99,
                                    max_wait_before_departure_anyway=1e9, soft_capacity_overload=1.5,
                                    nominal_capacity=10.0)
    state = init_fleet(net, cfg, batch_size=1, fleet_size=1, device=device)
    queue = torch.zeros(1, n_demand_stops(net), device=device)
    queue[0, 0] = 1000.0
    state, queue, boarded = step_informal(state, queue, net, cfg, gen)
    assert state.onboard[0, 0].item() == boarded[0].item()  # boarded passengers stay onboard once en route
    assert queue[0, 0].item() == 1000.0 - boarded[0].item()
    assert boarded[0].item() <= cfg.nominal_capacity * cfg.soft_capacity_overload + 1e-6
    assert boarded[0].item() > cfg.nominal_capacity  # exceeded nominal capacity


def test_multiple_vehicles_sharing_a_stop_do_not_double_book_passengers():
    cfg, net, device, gen = _setup(fleet_size=2, n_days=1, departure_threshold=1e9,
                                    max_wait_before_departure_anyway=1e9, nominal_capacity=10.0,
                                    soft_capacity_overload=1.0)
    state = init_fleet(net, cfg, batch_size=1, fleet_size=2, device=device)
    # force both vehicles onto the same corridor/stop (they already are: corridor 0 % 6, but
    # with 2 vehicles and 6 corridors they'd differ; pin explicitly)
    state.corridor[:] = 0
    queue = torch.zeros(1, n_demand_stops(net), device=device)
    queue[0, 0] = 8.0  # less than combined room (20) but more than one vehicle's capacity
    state, queue, boarded = step_informal(state, queue, net, cfg, gen)
    assert boarded[0].item() == 8.0
    assert queue[0, 0].item() == 0.0
    assert state.onboard.sum().item() == 8.0


def test_informal_rolling_pickup_serves_intermediate_junction_demand():
    """Regression test: informal vehicles must pick up passengers waiting at
    junctions they pass through en route, not only at their origin stop —
    otherwise intermediate demand would never be served except by rare
    deviations."""
    cfg, net, device, gen = _setup(
        fleet_size=1, n_days=1, n_corridors=1, n_junctions_per_corridor=4,
        departure_threshold=1e9, max_wait_before_departure_anyway=0.0,
        deviation_prob=0.0, nominal_capacity=50.0, soft_capacity_overload=1.0,
    )
    state = init_fleet(net, cfg, batch_size=1, fleet_size=1, device=device)
    queue = torch.zeros(1, n_demand_stops(net), device=device)
    # a passenger waiting at junction 1 (intermediate, not the vehicle's origin)
    queue[0, 1] = 5.0
    for _ in range(30):
        state, queue, boarded = step_informal(state, queue, net, cfg, gen)
        if state.onboard.sum().item() > 0:
            break
    assert state.onboard.sum().item() == 5.0
    assert queue[0, 1].item() == 0.0


def test_formal_fleet_departs_at_headway_regardless_of_load():
    cfg, net, device, gen = _setup(fleet_size=6, n_days=1, headway_minutes=5.0)
    state = formal_baseline.init_fleet(net, cfg, batch_size=1, fleet_size=6, device=device)
    queue = torch.zeros(1, n_demand_stops(net), device=device)  # no passengers ever
    departed_any = False
    for _ in range(10):
        state, queue, boarded = formal_baseline.step_formal(state, queue, net, cfg, gen)
        if (state.state == EN_ROUTE).any():
            departed_any = True
    assert departed_any


def test_formal_fleet_hard_capacity_leaves_excess_queued():
    cfg, net, device, gen = _setup(fleet_size=1, n_days=1, headway_minutes=1e9, hard_capacity=10.0)
    state = formal_baseline.init_fleet(net, cfg, batch_size=1, fleet_size=1, device=device)
    queue = torch.zeros(1, n_demand_stops(net), device=device)
    queue[0, 0] = 100.0
    state, queue, boarded = formal_baseline.step_formal(state, queue, net, cfg, gen)
    assert boarded[0].item() == 10.0
    assert queue[0, 0].item() == 90.0
