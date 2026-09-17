"""Synthetic road network: K radial corridors feeding a shared CBD hub.

Matches Lagos's well-documented radial structure (a handful of dominant
arterial routes into/out of Island/Mainland business districts, chronically
congested at predictable chokepoints such as bridges). Modelled as a graph:
nodes are junctions/stops, edges are road segments. Every corridor is a
simple path of `n_junctions_per_corridor` junctions whose innermost junction
is the *same* shared CBD node for all corridors — that shared node is where
routes "cross", and the edges feeding into it (bridges / CBD-adjacent
segments) are the ones flagged as bottlenecks with a steeper congestion curve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import networkx as nx
import torch

from .config import Config

BPR_POWER = 4.0  # standard Bureau of Public Roads congestion-function exponent


def _gaussian(t: float, peak: float, width: float) -> float:
    return math.exp(-0.5 * ((t - peak) / width) ** 2)


@dataclass
class Network:
    n_corridors: int
    n_junctions: int  # J: junctions per corridor, including the shared CBD terminus
    n_nodes: int
    n_edges: int
    cbd_node: int
    edge_from: torch.Tensor  # [E] long
    edge_to: torch.Tensor  # [E] long
    base_travel_time: torch.Tensor  # [E] float, minutes, free-flow
    congestion_sensitivity: torch.Tensor  # [E] float, BPR alpha
    capacity: torch.Tensor  # [E] float, vehicles at which congestion term == alpha
    is_bottleneck: torch.Tensor  # [E] bool

    @property
    def edges_per_corridor(self) -> int:
        return self.n_junctions - 1

    def edge_for_leg(self, corridor: torch.Tensor, local_j: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        """Edge id for a vehicle on `corridor`, currently at junction `local_j`,
        travelling in `direction` (0 = inbound toward CBD, 1 = outbound away).
        Valid only where local_j is a legal leg-start for that direction
        (inbound: 0..J-2; outbound: 1..J-1)."""
        epc = self.edges_per_corridor
        e_local = torch.where(direction == 0, local_j, local_j - 1)
        return corridor * epc + e_local

    def scale_congestion(self, factor: float) -> "Network":
        """Return a copy with every edge's congestion sensitivity scaled by
        `factor` — used by the §9b.2 congestion-severity sweep to ask how
        much of informal transit's advantage (or disadvantage) survives as
        bottleneck congestion is made more or less severe."""
        return Network(
            n_corridors=self.n_corridors,
            n_junctions=self.n_junctions,
            n_nodes=self.n_nodes,
            n_edges=self.n_edges,
            cbd_node=self.cbd_node,
            edge_from=self.edge_from,
            edge_to=self.edge_to,
            base_travel_time=self.base_travel_time,
            congestion_sensitivity=self.congestion_sensitivity * factor,
            capacity=self.capacity,
            is_bottleneck=self.is_bottleneck,
        )

    def to(self, device: torch.device) -> "Network":
        return Network(
            n_corridors=self.n_corridors,
            n_junctions=self.n_junctions,
            n_nodes=self.n_nodes,
            n_edges=self.n_edges,
            cbd_node=self.cbd_node,
            edge_from=self.edge_from.to(device),
            edge_to=self.edge_to.to(device),
            base_travel_time=self.base_travel_time.to(device),
            congestion_sensitivity=self.congestion_sensitivity.to(device),
            capacity=self.capacity.to(device),
            is_bottleneck=self.is_bottleneck.to(device),
        )

    def ambient_density(self, cfg: Config, t_hours: float) -> torch.Tensor:
        """Background (non-transit) road traffic per edge, shape [E]. The
        simulated transit fleet is far too small (tens of vehicles across
        dozens of edges) to ever load a bridge/CBD-adjacent edge up to a
        realistic congestion level by itself — real Lagos bottlenecks are
        congested by general traffic. This adds an exogenous, time-varying
        background density (higher at bottleneck edges, higher at rush
        hour) so the congestion function actually responds to time-of-day
        and to the §9b.2 congestion-severity sweep."""
        rush = max(
            _gaussian(t_hours, cfg.morning_peak_hour, cfg.peak_width_hours),
            _gaussian(t_hours, cfg.evening_peak_hour, cfg.peak_width_hours),
        )
        base = torch.where(
            self.is_bottleneck,
            torch.full_like(self.capacity, cfg.ambient_traffic_base * cfg.ambient_traffic_bottleneck_multiplier),
            torch.full_like(self.capacity, cfg.ambient_traffic_base),
        )
        peak_component = 1.0 + (cfg.ambient_traffic_peak_multiplier - 1.0) * rush
        return base * peak_component

    def travel_time(self, edge_idx: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        """BPR-style congestion function: t = t0 * (1 + alpha * (density/capacity)^4).

        `edge_idx` and `density` must be broadcastable tensors of edge ids and
        the current vehicle count on that edge, respectively.
        """
        t0 = self.base_travel_time[edge_idx]
        alpha = self.congestion_sensitivity[edge_idx]
        cap = self.capacity[edge_idx]
        ratio = (density / cap).clamp(min=0.0)
        return t0 * (1.0 + alpha * ratio.pow(BPR_POWER))


def generate_network(cfg: Config, gen: torch.Generator) -> Network:
    """Procedurally generate the corridor network. `gen` is a CPU
    torch.Generator (network generation is a one-off, not batched)."""
    K = cfg.n_corridors
    J = cfg.n_junctions_per_corridor
    if J < 2:
        raise ValueError("n_junctions_per_corridor must be >= 2")
    epc = J - 1  # edges per corridor
    n_edges = K * epc
    n_nodes = K * epc + 1
    cbd_node = n_nodes - 1

    corridor_idx = torch.arange(n_edges) // epc
    local_edge_idx = torch.arange(n_edges) % epc  # 0 = outermost edge

    edge_from = corridor_idx * epc + local_edge_idx
    edge_to = torch.where(
        local_edge_idx + 1 < epc,
        corridor_idx * epc + local_edge_idx + 1,
        torch.full((n_edges,), cbd_node, dtype=torch.long),
    )

    base_travel_time = 2.0 + 1.5 * torch.rand(n_edges, generator=gen)

    # Bottleneck edges are the ones closest to the CBD within each corridor
    # (bridges / CBD-adjacent segments), a documented chronic-congestion pattern.
    n_bottleneck = max(1, round(cfg.bottleneck_edges_frac * epc))
    hops_to_cbd = epc - local_edge_idx  # 1 = feeds directly into the CBD node
    is_bottleneck = hops_to_cbd <= n_bottleneck

    congestion_sensitivity = torch.where(
        is_bottleneck, torch.full((n_edges,), 4.0), torch.full((n_edges,), 0.8)
    )
    congestion_sensitivity = congestion_sensitivity * (0.85 + 0.3 * torch.rand(n_edges, generator=gen))

    capacity = torch.where(is_bottleneck, torch.full((n_edges,), 8.0), torch.full((n_edges,), 20.0))

    return Network(
        n_corridors=K,
        n_junctions=J,
        n_nodes=n_nodes,
        n_edges=n_edges,
        cbd_node=cbd_node,
        edge_from=edge_from,
        edge_to=edge_to,
        base_travel_time=base_travel_time,
        congestion_sensitivity=congestion_sensitivity,
        capacity=capacity,
        is_bottleneck=is_bottleneck,
    )


def to_networkx(net: Network) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(range(net.n_nodes))
    for e in range(net.n_edges):
        g.add_edge(
            int(net.edge_from[e]),
            int(net.edge_to[e]),
            base_travel_time=float(net.base_travel_time[e]),
            is_bottleneck=bool(net.is_bottleneck[e]),
        )
    return g
