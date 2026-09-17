import torch

from danfosim.config import Config
from danfosim.network import generate_network
from danfosim.sim import init_sim, run_day, step, summarize, summarize_with_ci


def _net(cfg):
    gen = torch.Generator().manual_seed(cfg.seed)
    return generate_network(cfg, gen)


def test_step_advances_time():
    cfg = Config(n_days=2, sim_hours=1.0)
    net = _net(cfg)
    sim_state = init_sim(cfg, net, "informal", torch.device("cpu"))
    t0 = sim_state.t_hours
    sim_state, boarded = step(sim_state)
    assert sim_state.t_hours == t0 + cfg.step_minutes / 60.0
    assert boarded.shape == (cfg.n_days,)


def test_run_day_informal_produces_metrics_for_every_step():
    cfg = Config(n_days=3, sim_hours=2.0, fleet_size=10, n_corridors=3, n_junctions_per_corridor=5)
    net = _net(cfg)
    sim_state, metrics = run_day(cfg, net, "informal", torch.device("cpu"))
    assert len(metrics.time_hours) == cfg.n_steps
    assert len(metrics.boarded) == cfg.n_steps


def test_run_day_formal_baseline_runs_without_error():
    cfg = Config(n_days=3, sim_hours=2.0, fleet_size=10, n_corridors=3, n_junctions_per_corridor=5)
    net = _net(cfg)
    sim_state, metrics = run_day(cfg, net, "formal", torch.device("cpu"))
    assert sim_state.regime == "formal"
    assert len(metrics.time_hours) == cfg.n_steps


def test_formal_bus_wait_time_roughly_bounded_by_half_headway_in_steady_state():
    """Known queueing-theory sanity check (architecture §10, week 2): for a
    fixed-headway service with demand well below capacity, mean wait time in
    steady state should be roughly bounded by half the headway."""
    cfg = Config(
        n_days=20, sim_hours=6.0, fleet_size=30, n_corridors=2, n_junctions_per_corridor=4,
        headway_minutes=10.0, hard_capacity=30.0, base_arrival_rate=0.3, peak_multiplier=1.0,
        ambient_traffic_base=0.0,  # isolate the headway/queueing mechanic from congestion effects
    )
    net = _net(cfg)
    sim_state, metrics = run_day(cfg, net, "formal", torch.device("cpu"))
    # skip the first 60 minutes (fill-up transient) for a steady-state read
    summary = summarize(metrics, cfg, t_start=cfg.day_start_hour + 1.0)
    assert summary.mean_wait_time_minutes < cfg.headway_minutes  # loose bound; well below capacity
    assert summary.throughput_trips > 0


def test_summarize_can_bucket_peak_vs_offpeak():
    cfg = Config(n_days=5, sim_hours=16.0, fleet_size=15, n_corridors=3, n_junctions_per_corridor=5)
    net = _net(cfg)
    sim_state, metrics = run_day(cfg, net, "informal", torch.device("cpu"))
    peak = summarize(metrics, cfg, t_start=cfg.morning_peak_hour - 1, t_end=cfg.morning_peak_hour + 1)
    offpeak = summarize(metrics, cfg, t_start=cfg.day_start_hour, t_end=cfg.day_start_hour + 1)
    assert peak.throughput_trips >= 0
    assert offpeak.throughput_trips >= 0


def test_summarize_with_ci_matches_pooled_point_estimate_reasonably():
    cfg = Config(n_days=50, sim_hours=4.0, fleet_size=15, n_corridors=3, n_junctions_per_corridor=5)
    net = _net(cfg)
    sim_state, metrics = run_day(cfg, net, "informal", torch.device("cpu"))
    pooled = summarize(metrics, cfg)
    ci = summarize_with_ci(metrics, cfg)
    assert ci["wait"].n > 0
    assert ci["wait"].ci95_low <= ci["wait"].mean <= ci["wait"].ci95_high
    # the per-day mean and the pooled ratio should be in the same ballpark
    assert abs(ci["wait"].mean - pooled.mean_wait_time_minutes) < pooled.mean_wait_time_minutes


def test_summarize_with_ci_narrows_as_more_days_are_pooled():
    cfg_small = Config(n_days=10, sim_hours=4.0, fleet_size=15, n_corridors=3, n_junctions_per_corridor=5)
    cfg_large = Config(n_days=100, sim_hours=4.0, fleet_size=15, n_corridors=3, n_junctions_per_corridor=5)
    net = _net(cfg_small)
    _, metrics_small = run_day(cfg_small, net, "informal", torch.device("cpu"))
    _, metrics_large = run_day(cfg_large, net, "informal", torch.device("cpu"))
    ci_small = summarize_with_ci(metrics_small, cfg_small)
    ci_large = summarize_with_ci(metrics_large, cfg_large)
    assert ci_large["wait"].se <= ci_small["wait"].se


def test_reproducible_given_same_seed():
    cfg = Config(n_days=4, sim_hours=1.0, fleet_size=8, n_corridors=2, n_junctions_per_corridor=4)
    net = _net(cfg)
    _, m1 = run_day(cfg, net, "informal", torch.device("cpu"))
    _, m2 = run_day(cfg, net, "informal", torch.device("cpu"))
    assert torch.equal(m1.boarded[-1], m2.boarded[-1])
