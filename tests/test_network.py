import torch

from danfosim.config import Config
from danfosim.network import generate_network, to_networkx


def _net(**overrides):
    cfg = Config(**overrides)
    gen = torch.Generator().manual_seed(cfg.seed)
    return cfg, generate_network(cfg, gen)


def test_node_and_edge_counts():
    cfg, net = _net(n_corridors=6, n_junctions_per_corridor=8)
    epc = 7
    assert net.n_edges == 6 * epc
    assert net.n_nodes == 6 * epc + 1
    assert net.cbd_node == net.n_nodes - 1


def test_all_corridors_terminate_at_shared_cbd_node():
    cfg, net = _net(n_corridors=4, n_junctions_per_corridor=5)
    epc = net.edges_per_corridor
    for k in range(net.n_corridors):
        last_edge = k * epc + (epc - 1)
        assert net.edge_to[last_edge].item() == net.cbd_node


def test_corridor_is_a_simple_chain():
    cfg, net = _net(n_corridors=3, n_junctions_per_corridor=6)
    epc = net.edges_per_corridor
    for k in range(net.n_corridors):
        for j in range(epc - 1):
            e = k * epc + j
            assert net.edge_to[e].item() == net.edge_from[k * epc + j + 1].item()


def test_bottleneck_edges_are_closest_to_cbd():
    cfg, net = _net(n_corridors=2, n_junctions_per_corridor=10, bottleneck_edges_frac=0.2)
    epc = net.edges_per_corridor
    for k in range(net.n_corridors):
        corridor_bottlenecks = net.is_bottleneck[k * epc : (k + 1) * epc]
        assert corridor_bottlenecks[-1].item() is True  # edge feeding into CBD
        # bottlenecks are a contiguous block ending at the CBD-adjacent edge
        n_bn = corridor_bottlenecks.sum().item()
        assert corridor_bottlenecks[epc - n_bn :].all()
        assert not corridor_bottlenecks[: epc - n_bn].any()


def test_travel_time_increases_with_density():
    cfg, net = _net()
    edge_idx = torch.tensor([0, 0, 0])
    density = torch.tensor([0.0, 5.0, 50.0])
    t = net.travel_time(edge_idx, density)
    assert t[0] < t[1] < t[2]


def test_travel_time_at_zero_density_is_base_travel_time():
    cfg, net = _net()
    edge_idx = torch.arange(net.n_edges)
    density = torch.zeros(net.n_edges)
    t = net.travel_time(edge_idx, density)
    assert torch.allclose(t, net.base_travel_time)


def test_networkx_conversion_is_connected_to_cbd():
    cfg, net = _net(n_corridors=5, n_junctions_per_corridor=4)
    g = to_networkx(net)
    assert g.number_of_nodes() == net.n_nodes
    assert g.number_of_edges() == net.n_edges
    for k in range(net.n_corridors):
        assert g.has_edge(k * net.edges_per_corridor + net.edges_per_corridor - 1, net.cbd_node)


def test_generation_is_reproducible_given_same_seed():
    cfg1, net1 = _net(seed=42)
    cfg2, net2 = _net(seed=42)
    assert torch.allclose(net1.base_travel_time, net2.base_travel_time)
    assert torch.equal(net1.is_bottleneck, net2.is_bottleneck)
