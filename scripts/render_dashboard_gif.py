"""Render a GIF preview of the interactive dashboard (docs/dashboard.html)
from the same trace data it uses, for embedding in the README — most
people reading a repo on GitHub won't spin up a local server to see
docs/dashboard.html actually move, so this gives them a preview.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

PAPER = "#f6f1e4"
INK = "#201c16"
INK_SOFT = "#5a5245"
ROAD = "#c3b8a1"
BOTTLENECK = "#c4401f"
DANFO = "#e8ac00"
DANFO_DEEP = "#9c6f00"
BRT = "#1c5d99"
BRT_DEEP = "#123f68"
QUEUE = "#c4401f"


def fmt_clock(t_hours: float) -> str:
    h = int(t_hours)
    m = round((t_hours - h) * 60)
    if m == 60:
        m = 0
        h += 1
    return f"{h:02d}:{m:02d}"


def draw_static(ax, net: dict, regime: str):
    nodes = {int(k): v for k, v in net["nodes"].items()}
    for e in net["edges"]:
        a, b = nodes[e["from"]], nodes[e["to"]]
        ax.plot([a[0], b[0]], [a[1], b[1]], color=BOTTLENECK if e["bottleneck"] else ROAD,
                linewidth=3.2 if e["bottleneck"] else 2.0, solid_capstyle="round", zorder=1)

    cbd = nodes[net["cbd_node"]]
    ax.scatter([cbd[0]], [cbd[1]], s=260, color=INK, zorder=3)
    ax.annotate("CBD", cbd, color=PAPER, ha="center", va="center", fontsize=7.5, fontweight="bold", zorder=4)

    title_color = DANFO_DEEP if regime == "informal" else BRT_DEEP
    label = "Informal — danfo/keke" if regime == "informal" else "Formal — fixed route"
    ax.set_title(label, color=title_color, fontsize=13, fontweight="bold", family="sans-serif", loc="left")

    ax.set_aspect("equal")
    ax.set_xlim(-480, 480)
    ax.set_ylim(-480, 480)
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, default="docs/dashboard-data")
    parser.add_argument("--out", type=str, default="docs/figures/dashboard.gif")
    parser.add_argument("--stride", type=int, default=2, help="use every Nth exported frame")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--preview-frame", type=int, default=None, help="save a single PNG frame instead of a GIF")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    net = json.loads((data_dir / "network.json").read_text())
    traces = {
        "informal": json.loads((data_dir / "trace_informal.json").read_text())["frames"][:: args.stride],
        "formal": json.loads((data_dir / "trace_formal.json").read_text())["frames"][:: args.stride],
    }
    n_frames = min(len(traces["informal"]), len(traces["formal"]))
    max_cap = net["config"]["nominal_capacity"]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.1), facecolor=PAPER)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.06, wspace=0.06)
    clock_text = fig.text(0.5, 0.94, "", ha="center", va="center", fontsize=17, fontweight="bold",
                           family="monospace", color=INK)
    fig.text(0.5, 0.985, "DanfoSim — one simulated day, real output", ha="center", va="top",
              fontsize=9.5, color=INK_SOFT, family="sans-serif")

    scatters = {}
    queue_scatters = {}
    for ax, regime in zip(axes, ("informal", "formal")):
        ax.set_facecolor(PAPER)
        draw_static(ax, net, regime)
        queue_scatters[regime] = ax.scatter([], [], s=[], color=QUEUE, alpha=0.55, zorder=2)
        color = DANFO if regime == "informal" else BRT
        edge = DANFO_DEEP if regime == "informal" else BRT_DEEP
        scatters[regime] = ax.scatter([], [], s=[], color=color, edgecolor=edge, linewidth=0.6, zorder=5)

    # scatter's `s` is marker area in points^2, not data units, so convert a
    # desired radius in the network's own coordinate space (matching the
    # dashboard.html SVG's radii) into points via the actual data-to-pixel
    # scale of this figure, measured once after layout is finalised.
    fig.canvas.draw()
    p0 = axes[0].transData.transform((0, 0))
    p1 = axes[0].transData.transform((1, 0))
    points_per_data_unit = abs(p1[0] - p0[0]) * 72.0 / fig.dpi

    def radius_to_area(r_data):
        return (r_data * points_per_data_unit) ** 2

    def update(i):
        t = traces["informal"][i]["t"]
        clock_text.set_text(fmt_clock(t))
        artists = [clock_text]
        for regime in ("informal", "formal"):
            frame = traces[regime][i]
            stop_xy = net["stop_xy"]
            qx = [p[0] for p in stop_xy]
            qy = [p[1] for p in stop_xy]
            qs = [radius_to_area(min(3 + q * 0.9, 22)) if q > 0.15 else 0.0 for q in frame["q"]]
            queue_scatters[regime].set_offsets(list(zip(qx, qy)))
            queue_scatters[regime].set_sizes(qs)

            vx = [v[0] for v in frame["v"]]
            vy = [v[1] for v in frame["v"]]
            sizes = []
            alphas = []
            for v in frame["v"]:
                waiting = v[2] == 0
                onboard = v[3]
                r = 5 if waiting else min(5 + (onboard / max_cap) * 6, 11)
                sizes.append(radius_to_area(r))
                alphas.append(0.4 if waiting else 1.0)
            scatters[regime].set_offsets(list(zip(vx, vy)))
            scatters[regime].set_sizes(sizes)
            scatters[regime].set_alpha(None)
            base_rgb = mcolors.to_rgb(DANFO if regime == "informal" else BRT)
            scatters[regime].set_facecolor([(*base_rgb, a) for a in alphas])
            artists.extend([queue_scatters[regime], scatters[regime]])
        return artists

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.preview_frame is not None:
        update(min(args.preview_frame, n_frames - 1))
        preview_path = out_path.with_suffix(".png")
        fig.savefig(preview_path, dpi=130, facecolor=PAPER)
        print(f"Wrote preview {preview_path}")
        return

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=args.fps))
    print(f"Wrote {out_path} ({n_frames} frames at {args.fps}fps)")


if __name__ == "__main__":
    main()
