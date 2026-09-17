import torch

from danfosim.config import Config
from danfosim.demand import (
    arrival_rate,
    demand_stop_id,
    n_demand_stops,
    sample_arrivals,
)
from danfosim.network import generate_network


def _net(**overrides):
    cfg = Config(**overrides)
    gen = torch.Generator().manual_seed(cfg.seed)
    return cfg, generate_network(cfg, gen)


def test_n_demand_stops_matches_corridors_times_junctions():
    cfg, net = _net(n_corridors=6, n_junctions_per_corridor=8)
    assert n_demand_stops(net) == 6 * 8


def test_demand_stop_id_is_unique_per_corridor_and_junction():
    cfg, net = _net(n_corridors=4, n_junctions_per_corridor=5)
    ids = set()
    for k in range(net.n_corridors):
        for j in range(net.n_junctions):
            sid = demand_stop_id(torch.tensor(k), torch.tensor(j), net.n_junctions).item()
            assert sid not in ids
            ids.add(sid)
    assert ids == set(range(n_demand_stops(net)))


def test_inbound_demand_peaks_in_the_morning():
    cfg, net = _net()
    off_peak = arrival_rate(net, cfg, t_hours=cfg.morning_peak_hour + 6.0)
    peak = arrival_rate(net, cfg, t_hours=cfg.morning_peak_hour)
    K, J = net.n_corridors, net.n_junctions
    # first inbound stop (corridor 0, local_j=0) should be far higher at peak
    assert peak[0] > off_peak[0]


def test_outbound_demand_peaks_in_the_evening_not_morning():
    cfg, net = _net()
    morning = arrival_rate(net, cfg, t_hours=cfg.morning_peak_hour)
    evening = arrival_rate(net, cfg, t_hours=cfg.evening_peak_hour)
    J = net.n_junctions
    outbound_stop = J - 1  # corridor 0's CBD outbound queue
    assert evening[outbound_stop] > morning[outbound_stop]


def test_farther_stops_have_higher_inbound_demand():
    cfg, net = _net(n_corridors=1, n_junctions_per_corridor=8)
    rate = arrival_rate(net, cfg, t_hours=cfg.morning_peak_hour)
    inbound_rates = rate[:-1]  # exclude the corridor's outbound CBD stop
    assert inbound_rates[0] > inbound_rates[-1]  # outermost > innermost


def test_sample_arrivals_shape_and_nonnegative():
    cfg, net = _net(n_days=10)
    gen = torch.Generator().manual_seed(0)
    arrivals = sample_arrivals(net, cfg, t_hours=7.5, batch_size=cfg.n_days, gen=gen, device=torch.device("cpu"))
    assert arrivals.shape == (cfg.n_days, n_demand_stops(net))
    assert (arrivals >= 0).all()


def test_sample_arrivals_batch_rows_are_independent_realisations():
    cfg, net = _net(n_days=200)
    gen = torch.Generator().manual_seed(0)
    arrivals = sample_arrivals(net, cfg, t_hours=7.5, batch_size=cfg.n_days, gen=gen, device=torch.device("cpu"))
    # not every simulated day should draw an identical arrival count everywhere
    assert not torch.all(arrivals == arrivals[0])
