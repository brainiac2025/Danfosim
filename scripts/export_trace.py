"""Export a single-day, per-vehicle simulation trace for visualization: the
2D layout of the network plus, for each sampled step, every vehicle's
screen position, state, and load, and every demand stop's queue length.
Used to build the interactive dashboard — not part of the analysis
pipeline itself.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from danfosim.config import Config
from danfosim.demand import n_demand_stops
from danfosim.network import Network, generate_network
from danfosim.sim import init_sim, step


def layout_network(net: Network) -> dict:
    """Radial layout: CBD at the origin, each corridor a spoke, junctions
    spaced by distance-from-CBD along the spoke."""
    K, J = net.n_corridors, net.n_junctions
    epc = net.edges_per_corridor
    spacing = 60.0

    node_xy: dict[int, list[float]] = {net.cbd_node: [0.0, 0.0]}
    for k in range(K):
        angle = 2 * math.pi * k / K - math.pi / 2
        for j in range(epc):
            node_id = k * epc + j
            radius = (epc - j) * spacing
            node_xy[node_id] = [radius * math.cos(angle), radius * math.sin(angle)]

    edges = [
        {"from": int(net.edge_from[e]), "to": int(net.edge_to[e]), "bottleneck": bool(net.is_bottleneck[e])}
        for e in range(net.n_edges)
    ]

    # Demand-stop positions: inbound stops share their junction's node;
    # each corridor's outbound (CBD) queue gets a small satellite offset
    # near the CBD hub so it's visible as its own marker.
    stop_xy: list[list[float]] = []
    for k in range(K):
        angle = 2 * math.pi * k / K - math.pi / 2
        for j in range(J):
            if j < J - 1:
                stop_xy.append(node_xy[k * epc + j])
            else:
                r = 0.4 * spacing
                stop_xy.append([r * math.cos(angle), r * math.sin(angle)])

    return {
        "nodes": node_xy,
        "edges": edges,
        "cbd_node": net.cbd_node,
        "stop_xy": stop_xy,
        "spacing": spacing,
    }


def edge_xy_lookup(net: Network, node_xy: dict[int, list[float]]) -> dict[tuple[int, int, int], tuple[list, list]]:
    """(corridor, local_j, direction) -> (from_xy, to_xy) for every legal
    leg-start, so a vehicle's screen position can be interpolated along its
    current edge each frame."""
    K, J = net.n_corridors, net.n_junctions
    epc = net.edges_per_corridor
    lookup = {}
    for k in range(K):
        for j in range(epc):
            e = k * epc + j
            frm, to = int(net.edge_from[e]), int(net.edge_to[e])
            lookup[(k, j, 0)] = (node_xy[frm], node_xy[to])  # inbound
            lookup[(k, j + 1, 1)] = (node_xy[to], node_xy[frm])  # outbound
    return lookup


def export_regime(cfg: Config, net: Network, regime: str, device: torch.device, sample_every: int, lookup) -> dict:
    sim_state = init_sim(cfg, net, regime, device)
    frames = []
    for i in range(cfg.n_steps):
        t_hours = sim_state.t_hours
        sim_state, boarded = step(sim_state)
        if i % sample_every != 0:
            continue

        fleet = sim_state.fleet
        corridor = fleet.corridor[0].tolist()
        direction = fleet.direction[0].tolist()
        cur_j = fleet.cur_j[0].tolist()
        state = fleet.state[0].tolist()
        onboard = fleet.onboard[0].tolist()
        remaining = fleet.remaining_travel_time[0].tolist()

        vehicles = []
        for k, d, j, s, ob, rem in zip(corridor, direction, cur_j, state, onboard, remaining):
            if s == 0:  # WAITING: sit at the junction — the "from" endpoint
                # of the leg that would start here (valid for both directions
                # since a waiting vehicle's cur_j is always a legal leg-start)
                xy = lookup[(k, j, d)][0]
            else:
                base_key = (k, j, d)
                frm_xy, to_xy = lookup[base_key]
                # progress uses base travel time as a rough (uncongested) scale
                edge_epc = net.edges_per_corridor
                e_local = j if d == 0 else j - 1
                edge_id = k * edge_epc + e_local
                base_tt = float(net.base_travel_time[edge_id])
                progress = min(max(1.0 - rem / max(base_tt, 1e-6), 0.0), 1.0)
                xy = [frm_xy[0] + (to_xy[0] - frm_xy[0]) * progress, frm_xy[1] + (to_xy[1] - frm_xy[1]) * progress]
            vehicles.append([round(xy[0], 1), round(xy[1], 1), s, round(ob, 1)])

        queue = sim_state.queue[0].tolist()
        trips_so_far = float(fleet.trips_completed[0].sum().item())
        frames.append({"t": round(t_hours, 3), "v": vehicles, "q": [round(x, 1) for x in queue], "trips": round(trips_so_far)})

    return {"regime": regime, "frames": frames}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-every", type=int, default=3, help="sample one frame every N simulated minutes")
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    cfg = Config(seed=args.seed, device=args.device)
    device = cfg.resolved_device()
    gen_net = torch.Generator().manual_seed(cfg.seed)
    net = generate_network(cfg, gen_net)

    layout = layout_network(net)
    lookup = edge_xy_lookup(net, layout["nodes"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "network.json", "w") as f:
        json.dump(
            {
                "nodes": layout["nodes"],
                "edges": layout["edges"],
                "cbd_node": layout["cbd_node"],
                "stop_xy": layout["stop_xy"],
                "n_corridors": net.n_corridors,
                "n_junctions": net.n_junctions,
                "config": {
                    "nominal_capacity": cfg.nominal_capacity,
                    "hard_capacity": cfg.hard_capacity,
                    "fleet_size": cfg.fleet_size,
                    "day_start_hour": cfg.day_start_hour,
                    "sim_hours": cfg.sim_hours,
                    "morning_peak_hour": cfg.morning_peak_hour,
                    "evening_peak_hour": cfg.evening_peak_hour,
                    "peak_width_hours": cfg.peak_width_hours,
                },
            },
            f,
        )
    print(f"Wrote {out_dir / 'network.json'}")

    for regime in ("informal", "formal"):
        data = export_regime(cfg, net, regime, device, args.sample_every, lookup)
        path = out_dir / f"trace_{regime}.json"
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"Wrote {path}  ({len(data['frames'])} frames)")


if __name__ == "__main__":
    main()
