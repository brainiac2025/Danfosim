# DanfoSim

A multi-agent simulation of Lagos's informal transit system: danfo minibuses and keke napep tricycles that don't run on a timetable, don't have official stops the way a proper bus route does, and are driven by people trying to make money that day, not hit a schedule. Almost all transit simulation research assumes fixed routes and fixed headways, because that's how the systems most researchers study actually work. This one doesn't, on purpose. No real trip data went into it. The agents, routes, and demand are generated from rules I calibrated against publicly documented facts about how the system operates, and I've tried to be upfront about which parts of that are backed by something real and which parts I just picked because they produced sane behavior (see [`docs/CALIBRATION.md`](docs/CALIBRATION.md)).

**The question I actually wanted an answer to**: does that flexibility buy you anything? Give Lagos a fixed-route, fixed-headway system instead, same roads, same vehicles, same passengers, and does it get better, worse, or does it just depend on the time of day? Run it yourself:

```bash
python scripts/run_comparison.py --n-days 200
python scripts/plot_results.py
```

The full write-up is in [`docs/REPORT.md`](docs/REPORT.md). There's also an interactive page where you can watch a simulated day actually play out, vehicles moving, queues building at stops, both regimes side by side. It's [`docs/dashboard.html`](docs/dashboard.html), and because it fetches its own data files it needs to be served rather than double-clicked:

```bash
cd docs && python -m http.server
# then open http://localhost:8000/dashboard.html
```

## Why nobody's really done this comparison before

I went looking for an existing benchmark that runs informal and formal transit head-to-head on the same network and couldn't find one. Formal transit microsimulation is a mature field. Work on informal transit systems exists too, but most of it just describes how these systems behave rather than building something you could run an experiment on. That's the gap this fills, and it's also why the calibration is thinner than I'd like: there's no ridership data or GPS traces to check any of this against. `docs/CALIBRATION.md` says exactly which numbers are grounded in something documented and which ones I tuned until the output stopped looking absurd.

## Install

```bash
python -m venv .venv
source .venv/Scripts/activate  # or .venv/bin/activate on Linux/Mac
pip install -r requirements-lock.txt
pip install -e . --no-deps
```

Needs Python 3.11+. `requirements-lock.txt` pins exactly what I developed and tested against (CPU-only PyTorch, since I don't have a CUDA box handy). If you're running this on actual GPU hardware, install a matching CUDA build of torch first, or skip the lock file and run `pip install -e ".[dev]"` for whatever's current.

## Tests

```bash
pytest
```

35 tests, a couple of seconds. They check the mechanics are right (passengers aren't created or destroyed, a formal bus's wait time roughly matches the queueing-theory bound it should, that kind of thing). They don't check whether the model says anything true about Lagos.

## Layout

```
danfosim/
├── network.py           # the radial road network
├── demand.py             # passenger arrivals, morning/evening peaks
├── vehicle.py              # danfo/keke agent: threshold departure, deviation
├── formal_baseline.py        # fixed-route, fixed-headway comparison agent
├── sim.py                      # batched sim loop + metrics
├── dispatch_policy.py            # learned dispatch policy
└── config.py                       # every hyperparameter, one place

scripts/
├── run_comparison.py    # the headline informal-vs-formal experiment
├── run_sensitivity.py     # threshold and congestion sweeps
├── run_dispatch_train.py    # trains the learned policy, measures departure regularity
├── export_trace.py            # exports the data behind the dashboard
└── plot_results.py              # the comparison figure

docs/CALIBRATION.md     # every parameter, cited or flagged as a guess
docs/REPORT.md          # the write-up
docs/dashboard.html     # watch a simulated day
tests/                  # pytest suite
```

## How it actually works

The network is six corridors radiating out to a shared CBD node, roughly the shape of Lagos's road geography: a handful of arterial routes converging on the business districts, with the well-known chokepoints (the bridges, basically) sitting right where each corridor feeds into the CBD. Those edges get a steeper congestion curve.

Passengers arrive at each stop following a time-varying Poisson process, with a morning peak heading into the CBD and an evening peak heading back out. This is the one part of the demand model I'm genuinely confident calibrating, since the AM/PM commute pattern is about as well documented as anything gets.

Informal vehicles wait and pick up passengers until a load threshold or a max-wait timer is hit, then go. While en route there's a small chance of cutting the trip short at whatever junction it's passing, if that looks like a better move: the "drop the passengers and go chase better demand" behavior that's pretty central to how danfo drivers actually operate. Otherwise it picks up anyone waiting at junctions it passes without actually stopping, because that's how curbside hailing works.

The formal baseline is fixed route, fixed headway from the terminus, a short dwell at every stop along the way, hard capacity. Nothing clever about it on purpose, it's the comparison point.

The sim loop runs `n_days` simulated days at once as batched tensor ops, no Python loop over individual vehicles or passengers, so a full day for both regimes takes a few seconds on a CPU.

The learned dispatch policy (`dispatch_policy.py`) replaces the fixed threshold rule with a small policy trained with REINFORCE to maximize something like revenue per trip minus a cost for making people wait, specifically so it can't just learn to leave the moment anyone's aboard. `run_dispatch_train.py` asks whether that produces more regular departures as more vehicles compete for the same passengers. In this model, no: it gets more erratic, not less.

## Reproducibility

Everything takes a `--seed`. The network and all the randomness downstream of it come from seeded `torch.Generator`s, so the same seed gets you the same run. `--n-days` is the parallel batch size: more days means tighter confidence intervals and a longer wall clock. `--device cpu`/`cuda` picks the hardware.
