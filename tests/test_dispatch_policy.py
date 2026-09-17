import torch

from danfosim.config import Config
from danfosim.demand import n_demand_stops
from danfosim.dispatch_policy import DispatchPolicy, make_features, step_learned
from danfosim.network import generate_network
from danfosim.vehicle import init_fleet


def _setup(**overrides):
    cfg = Config(**overrides)
    gen_net = torch.Generator().manual_seed(cfg.seed)
    net = generate_network(cfg, gen_net)
    device = torch.device("cpu")
    gen = torch.Generator(device=device).manual_seed(cfg.seed + 1)
    return cfg, net, device, gen


def test_step_learned_runs_and_returns_expected_shapes():
    cfg, net, device, gen = _setup(fleet_size=6, n_days=2, n_corridors=2, n_junctions_per_corridor=4)
    fleet = init_fleet(net, cfg, batch_size=2, fleet_size=6, device=device)
    queue = torch.zeros(2, n_demand_stops(net), device=device)
    policy = DispatchPolicy(hidden=8)

    fleet, queue, boarded, depart_now, policy_depart, logprob, entropy, reward = step_learned(
        fleet, queue, net, cfg, gen, policy, t_hours=cfg.day_start_hour
    )
    assert fleet.corridor.shape == (2, 6)
    assert depart_now.shape == (2, 6)
    assert policy_depart.shape == (2, 6)


def test_forced_departure_overrides_policy_at_max_wait():
    cfg, net, device, gen = _setup(fleet_size=1, n_days=1, max_wait_before_departure_anyway=2.0, base_arrival_rate=0.0)
    fleet = init_fleet(net, cfg, batch_size=1, fleet_size=1, device=device)
    queue = torch.zeros(1, n_demand_stops(net), device=device)
    policy = DispatchPolicy(hidden=8)
    # bias the policy heavily toward "never depart" so only the forced fallback can trigger
    with torch.no_grad():
        for p in policy.parameters():
            p.zero_()
        policy.net[-1].bias.fill_(-100.0)

    forced_departure_seen = False
    policy_departure_seen = False
    for _ in range(3):
        fleet, queue, boarded, depart_now, policy_depart, logprob, entropy, reward = step_learned(
            fleet, queue, net, cfg, gen, policy, t_hours=cfg.day_start_hour
        )
        forced_departure_seen |= bool(depart_now[0, 0].item())
        policy_departure_seen |= bool(policy_depart[0, 0].item())
    assert forced_departure_seen  # the max-wait safety fallback still fires
    assert not policy_departure_seen  # but never as the policy's own choice


def test_gradient_flows_from_reinforce_loss_to_policy_parameters():
    cfg, net, device, gen = _setup(fleet_size=20, n_days=4, base_arrival_rate=5.0)
    fleet = init_fleet(net, cfg, batch_size=4, fleet_size=20, device=device)
    queue = torch.zeros(4, n_demand_stops(net), device=device)
    policy = DispatchPolicy(hidden=8)

    loss = None
    for _ in range(20):
        fleet, queue, boarded, depart_now, policy_depart, logprob, entropy, reward = step_learned(
            fleet, queue, net, cfg, gen, policy, t_hours=cfg.day_start_hour
        )
        if policy_depart.any():
            r = reward[policy_depart].detach()
            loss = -(r * logprob[policy_depart]).mean()
            break
    assert loss is not None
    policy.zero_grad()
    loss.backward()
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(g.abs().sum().item() > 0 for g in grads)
