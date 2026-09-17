"""§9b.1-2 — supporting sensitivity experiments:

  threshold   sweep `departure_threshold` and `max_wait_before_departure_anyway`,
              showing the wait-time/utilisation tradeoff informal operators are
              implicitly navigating.
  congestion  sweep bottleneck congestion severity and report how much of
              informal transit's advantage over formal (§9a) shrinks as
              travel time comes to dominate wait time.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from danfosim.config import Config
from danfosim.network import generate_network
from danfosim.sim import run_day, summarize


def sweep_threshold(cfg: Config, net, device, thresholds, max_waits) -> list[dict]:
    results = []
    for threshold in thresholds:
        for max_wait in max_waits:
            run_cfg = replace(cfg, departure_threshold=threshold, max_wait_before_departure_anyway=max_wait)
            _, metrics = run_day(run_cfg, net, "informal", device)
            summary = summarize(metrics, run_cfg)
            row = {"departure_threshold": threshold, "max_wait_before_departure_anyway": max_wait, **asdict(summary)}
            results.append(row)
            print(
                f"  threshold={threshold:.2f}  max_wait={max_wait:5.1f}min  "
                f"wait={summary.mean_wait_time_minutes:7.2f}min  util={summary.mean_utilisation:5.2f}  "
                f"trips={summary.throughput_trips:7.0f}"
            )
    return results


def sweep_congestion(cfg: Config, net, device, multipliers) -> list[dict]:
    results = []
    for factor in multipliers:
        scaled_net = net.scale_congestion(factor)
        row = {"congestion_multiplier": factor}
        for regime in ("informal", "formal"):
            _, metrics = run_day(cfg, scaled_net, regime, device)
            summary = summarize(metrics, cfg)
            row[regime] = asdict(summary)
        advantage = row["formal"]["mean_wait_time_minutes"] - row["informal"]["mean_wait_time_minutes"]
        row["informal_wait_advantage_minutes"] = advantage
        results.append(row)
        print(
            f"  congestion x{factor:.2f}  informal wait={row['informal']['mean_wait_time_minutes']:7.2f}min  "
            f"formal wait={row['formal']['mean_wait_time_minutes']:7.2f}min  advantage={advantage:+7.2f}min"
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["threshold", "congestion"])
    parser.add_argument("--n-days", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.n_days is not None:
        cfg.n_days = args.n_days
    if args.device is not None:
        cfg.device = args.device
    if args.seed is not None:
        cfg.seed = args.seed

    device = cfg.resolved_device()
    gen_net = torch.Generator().manual_seed(cfg.seed)
    net = generate_network(cfg, gen_net)

    if args.mode == "threshold":
        print(f"§9b.1 threshold sensitivity — {cfg.n_days} simulated days on {device}")
        thresholds = [0.5, 0.65, 0.75, 0.9, 1.0]
        max_waits = [4.0, 8.0, 15.0]
        results = sweep_threshold(cfg, net, device, thresholds, max_waits)
        out_path = Path(args.out or "runs/sensitivity_threshold.json")
    else:
        print(f"§9b.2 congestion sensitivity — {cfg.n_days} simulated days on {device}")
        multipliers = [0.25, 0.5, 1.0, 2.0, 4.0]
        results = sweep_congestion(cfg, net, device, multipliers)
        out_path = Path(args.out or "runs/sensitivity_congestion.json")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"config": vars(cfg), "mode": args.mode, "results": results}, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
