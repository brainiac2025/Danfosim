"""§9a — the headline experiment: informal vs. formal transit on the
identical network and demand, fleet size held equal, across a full simulated
day with the morning/evening peak structure. Reports mean wait time, travel
time, and throughput for each regime, broken out by time-of-day.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from danfosim.config import Config
from danfosim.network import generate_network
from danfosim.sim import run_day, summarize, summarize_with_ci


def print_row(label: str, summary, ci) -> None:
    wait_ci = ci["wait"]
    print(
        f"  {label:<12} wait={summary.mean_wait_time_minutes:7.2f} min "
        f"[95% CI {wait_ci.ci95_low:6.2f}, {wait_ci.ci95_high:6.2f}, n={wait_ci.n:3d} days]  "
        f"travel={summary.mean_travel_time_minutes:7.2f} min  "
        f"util={summary.mean_utilisation:5.2f}  trips={summary.throughput_trips:8.0f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-days", type=int, default=None, help="override Config.n_days (parallel batch size)")
    parser.add_argument("--device", type=str, default=None, help="override Config.device ('cuda' or 'cpu')")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=str, default="runs/comparison.json")
    args = parser.parse_args()

    cfg = Config()
    if args.n_days is not None:
        cfg.n_days = args.n_days
    if args.device is not None:
        cfg.device = args.device
    if args.seed is not None:
        cfg.seed = args.seed

    device = cfg.resolved_device()
    print(f"Running §9a comparison: {cfg.n_days} simulated days on {device}, fleet_size={cfg.fleet_size} per regime.")

    gen_net = torch.Generator().manual_seed(cfg.seed)
    net = generate_network(cfg, gen_net)

    morning_lo, morning_hi = cfg.morning_peak_hour - 1.0, cfg.morning_peak_hour + 1.0
    evening_lo, evening_hi = cfg.evening_peak_hour - 1.0, cfg.evening_peak_hour + 1.0
    offpeak_lo, offpeak_hi = cfg.day_start_hour, min(morning_lo, cfg.day_start_hour + cfg.sim_hours)

    results = {}
    for regime in ("informal", "formal"):
        _, metrics = run_day(cfg, net, regime, device)

        windows = {
            "overall": (None, None),
            "morning_peak": (morning_lo, morning_hi),
            "evening_peak": (evening_lo, evening_hi),
        }
        if offpeak_hi > offpeak_lo:
            windows["off_peak"] = (offpeak_lo, offpeak_hi)

        print(f"\n=== {regime} ===")
        bucket_results = {}
        for label, (t_start, t_end) in windows.items():
            summary = summarize(metrics, cfg, t_start=t_start, t_end=t_end)
            ci = summarize_with_ci(metrics, cfg, t_start=t_start, t_end=t_end)
            print_row(label.replace("_", " "), summary, ci)
            bucket_results[label] = {**asdict(summary), "wait_time_ci95": [ci["wait"].ci95_low, ci["wait"].ci95_high], "wait_time_n_days": ci["wait"].n}

        results[regime] = bucket_results

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"config": vars(cfg), "results": results}, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
