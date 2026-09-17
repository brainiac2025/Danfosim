"""Plot the §9a informal-vs-formal comparison from a `run_comparison.py`
JSON output: grouped bars for wait time, travel time, and utilisation,
broken out by time-of-day bucket.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_metric(ax, data: dict, metric: str, ylabel: str, title: str) -> None:
    buckets = ["off_peak", "morning_peak", "evening_peak", "overall"]
    labels = ["off-peak", "morning peak", "evening peak", "overall"]
    regimes = ["informal", "formal"]
    colors = {"informal": "#d9822b", "formal": "#2b6cd9"}

    x = np.arange(len(buckets))
    width = 0.35
    for i, regime in enumerate(regimes):
        values = []
        for bucket in buckets:
            entry = data["results"][regime].get(bucket)
            values.append(entry[metric] if entry is not None else float("nan"))
        offset = (i - 0.5) * width
        ax.bar(x + offset, values, width, label=regime, color=colors[regime])

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", type=str, default="runs/comparison.json")
    parser.add_argument("--out", type=str, default="runs/comparison.png")
    args = parser.parse_args()

    data = load_results(Path(args.in_path))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    plot_metric(axes[0], data, "mean_wait_time_minutes", "minutes", "Mean passenger wait time")
    plot_metric(axes[1], data, "mean_travel_time_minutes", "minutes", "Mean in-vehicle travel time")
    plot_metric(axes[2], data, "mean_utilisation", "fraction of capacity", "Mean vehicle utilisation")
    fig.suptitle("DanfoSim §9a — informal vs. formal transit, same network & demand")
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
