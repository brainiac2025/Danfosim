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


def plot_metric(ax, data: dict, metric: str, ylabel: str, title: str, show_ci: bool = False) -> None:
    buckets = ["off_peak", "morning_peak", "evening_peak", "overall"]
    labels = ["off-peak", "morning peak", "evening peak", "overall"]
    regimes = ["informal", "formal"]
    colors = {"informal": "#d9822b", "formal": "#2b6cd9"}

    x = np.arange(len(buckets))
    width = 0.35
    for i, regime in enumerate(regimes):
        values, err_low, err_high = [], [], []
        for bucket in buckets:
            entry = data["results"][regime].get(bucket)
            value = entry[metric] if entry is not None else float("nan")
            values.append(value)
            if show_ci and entry is not None and "wait_time_ci95" in entry:
                lo, hi = entry["wait_time_ci95"]
                err_low.append(max(value - lo, 0.0))
                err_high.append(max(hi - value, 0.0))
            else:
                err_low.append(0.0)
                err_high.append(0.0)
        offset = (i - 0.5) * width
        yerr = [err_low, err_high] if show_ci else None
        ax.bar(x + offset, values, width, label=regime, color=colors[regime], yerr=yerr, capsize=3, ecolor="black")

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
    plot_metric(axes[0], data, "mean_wait_time_minutes", "minutes", "Mean passenger wait time (95% CI)", show_ci=True)
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
