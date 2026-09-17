# DanfoSim

A multi-agent simulation of Lagos's informal, route-negotiated public
transport (danfo minibuses, keke napep tricycles) — a transport mode with no
fixed timetables, driver-negotiated stops, and route/fare dynamics
fundamentally unlike the scheduled, fixed-route transit that almost all
transit-simulation research assumes. No real trip data is used: agents,
routes, and demand are procedurally generated from rules calibrated to
publicly documented features of how the system operates (see
[`docs/CALIBRATION.md`](docs/CALIBRATION.md) for exactly which rules are
"documented pattern" vs. "reasonable assumption").

**The central question**: does informal transit's flexibility — departing
when full rather than on a schedule, deviating toward local demand — help or
hurt relative to a fixed-route, fixed-headway formal system, on the *same*
road network and demand? Run the comparison yourself:

```bash
python scripts/run_comparison.py --n-days 200
python scripts/plot_results.py
```

## Why this is hard to get right elsewhere

There is no substantial existing benchmark of informal-vs-formal transit
simulated head-to-head on the same network. Formal transit microsimulation
is a mature field; informal-transit-specific simulation is much thinner and
mostly qualitative. That makes this comparison's contribution genuinely
novel, but also means its calibration is more exposed to challenge than a
project with a mature reference model — read
[`docs/CALIBRATION.md`](docs/CALIBRATION.md) for the honest accounting of
which numbers are cited-pattern vs. assumption.

## Install

```bash
python -m venv .venv
source .venv/Scripts/activate  # or .venv/bin/activate on Linux/Mac
pip install -r requirements-lock.txt
pip install -e . --no-deps
```

Requires Python 3.11+. `requirements-lock.txt` pins the exact versions this
project was developed and tested against (CPU-only PyTorch). On the target
GPU hardware, install a matching CUDA build of `torch` instead before
running the lock file, or just `pip install -e ".[dev]"` for unpinned
latest-compatible versions.

## Run the tests

```bash
pytest
```

## Project layout

```
danfosim/
├── network.py           # synthetic radial-corridor road network (§3)
├── demand.py             # bimodal Poisson passenger arrivals (§4)
├── vehicle.py              # informal (danfo/keke) agent: threshold departure, deviation (§5)
├── formal_baseline.py        # fixed-route, fixed-headway comparison agent (§6)
├── sim.py                      # batched multi-agent step loop + metrics (§6-7)
├── dispatch_policy.py            # learned route-adaptation policy (§8)
└── config.py                       # single dataclass of hyperparameters

scripts/
├── run_comparison.py    # §9a: informal vs. formal, the headline experiment
├── run_dispatch_train.py  # §8/§9b.3: train the learned policy, measure departure regularity
└── plot_results.py          # renders the §9a comparison figure

docs/CALIBRATION.md     # every behavioural rule, cited or flagged as assumption
tests/                  # pytest suite for network, demand, vehicle, sim, dispatch_policy
```

## Method summary

- **Network** (`network.py`): `n_corridors` radial corridors, each a simple
  chain of junctions, all sharing one CBD hub node — matching Lagos's
  well-documented radial structure. Edges nearest the CBD are flagged as
  bottlenecks (bridges / CBD-adjacent congestion) with a steeper BPR-style
  congestion curve.
- **Demand** (`demand.py`): time-varying Poisson arrivals per stop, with a
  bimodal peak (morning inbound to the CBD, evening outbound) — the single
  most robustly documented fact about Lagos commute patterns.
- **Informal vehicles** (`vehicle.py`): a state machine — wait and
  accumulate passengers until a load threshold (or a max-wait cap) is met,
  then depart; en route, a small probability of ending the trip early at an
  intermediate junction to chase higher local demand ("dropping" short of
  the nominal terminus); otherwise a zero-dwell rolling curbside pickup at
  junctions passed along the way.
- **Formal baseline** (`formal_baseline.py`): fixed route, fixed headway
  from the terminus, a brief load-independent dwell at every intermediate
  junction, and a hard capacity cap (excess demand queues for the next bus).
- **Simulation** (`sim.py`): `n_days` independent simulated days run as
  batched tensor state (`[n_days, fleet_size]` / `[n_days, n_edges]` /
  `[n_days, n_demand_stops]`) — no Python loop over individual vehicles or
  passengers.
- **Learned dispatch policy** (`dispatch_policy.py`, §8): a small
  contextual-bandit policy replaces the fixed threshold rule, trained via
  REINFORCE to maximise a revenue-like reward (occupancy carried, minus a
  waiting-time cost) — the tension needed to avoid collapsing to
  "always depart immediately". `run_dispatch_train.py` also measures whether
  the learned departure pattern becomes more regular (lower inter-departure
  variance) as fleet size grows.

## Reproducibility

Every run takes a `--seed` (default from `Config.seed = 0`); the network and
all demand/agent randomness are drawn from seeded `torch.Generator`s.
`--n-days` controls the parallel batch size (statistical confidence vs. wall
clock); `--device cpu`/`cuda` controls hardware.
