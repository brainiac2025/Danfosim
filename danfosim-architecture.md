# DanfoSim — Agent-Based Simulation of Lagos Informal Transit

A multi-agent simulation of Lagos's informal, route-negotiated public transport
(danfo minibuses, keke napep tricycles) — a transport mode with no fixed
timetables, driver-negotiated stops, and route/fare dynamics fundamentally
unlike scheduled Western transit systems that transport-simulation research
almost universally assumes. No real trip data needed — agents, routes, and
demand are procedurally generated from rules you define and calibrate to
publicly documented features of how the system operates. Target hardware:
Legion 7 Pro (16 GB GPU, 64 GB RAM) — this project is agent-count-bound rather
than model-size-bound; the GPU's role is running large simulated populations,
not big networks.

**Deliverables**
1. Public repo with tests, reproducible simulation, pinned dependencies.
2. A calibration document grounding every behavioural rule in a cited source
   describing how Lagos informal transit actually operates.
3. A congestion/wait-time benchmark comparing informal (negotiated,
   no-fixed-schedule) vs. formalised (fixed-route, fixed-headway) operation on
   the *same* road network and demand — the project's central question.
4. A learned dispatch/route-adaptation policy for an informal operator, and a
   measurement of whether it converges toward or away from formal-transit
   behaviour under pressure to reduce wait times.
5. 4–6 page write-up centred on the informal-vs-formal comparison.

---

## 0. Honesty about novelty and risk

Say this plainly, more than for the other portfolio projects: **there is no
substantial existing benchmark of "informal vs formal transit" simulated
head-to-head on the same network**, which makes this the most genuinely
open-ended of the group — but also the one with the least publicly documented
ground truth to calibrate against. Formal transit simulation (fixed-route
agent-based models, MATSim-style microsimulation) is an enormous, mature field;
informal-transit-specific simulation work exists but is much thinner and mostly
qualitative/policy-focused rather than a reusable computational model. That
means this project's contribution is more clearly real (the comparison itself
is close to unaddressed) but its calibration is also more exposed to challenge
than NaijaGrid's — be explicit in the write-up about which behavioural rules
are "documented pattern" versus "reasonable assumption," and don't overstate
confidence in the exact numbers.

---

## 1. What makes informal transit structurally different (the model's core)

Three mechanisms, not present in standard fixed-route transit models, are what
this simulation needs to get right — everything else is standard traffic/queue
simulation:

- **No fixed schedule, threshold-based departure**: a danfo doesn't leave at a
  scheduled time; it leaves when it's sufficiently full (a load-threshold rule),
  so wait time at a stop is a function of arrival-rate-vs-threshold, not a
  published headway.
  
  - **Route flexibility under demand**: drivers can deviate, wait at
  higher-demand junctions, or informally split a route into shorter segments
  ("dropping" passengers short of the nominal terminus and picking up new ones)
  when that increases trips-per-hour revenue — an economic optimization real
  formal-transit models don't include because formal drivers don't have this
  freedom.
- **Fare and capacity elasticity**: fares can vary informally with
  congestion/demand/time-of-day (a form of surge pricing that isn't centrally
  set), and "capacity" is soft — a danfo can carry more than its nominal seats
  under pressure, unlike a formal bus's hard capacity limit.

---

## 2. Repository layout

```
danfosim/
├── README.md                    # headline comparison figure at the top
├── pyproject.toml                # python 3.11, torch, numpy, networkx, pytest
├── danfosim/
│   ├── network.py                 # synthetic road network + corridor demand (§3)
│   ├── demand.py                   # passenger arrival generation (§4)
│   ├── vehicle.py                   # danfo/keke agent: threshold departure, route logic (§5)
│   ├── formal_baseline.py            # fixed-route, fixed-headway comparison agent (§6)
│   ├── sim.py                          # batched multi-agent step loop, GPU-vectorised (§7)
│   ├── dispatch_policy.py               # learned route-adaptation policy (§8)
│   └── config.py                         # single dataclass of hyperparameters
├── scripts/
│   ├── run_comparison.py             # §9a: informal vs formal, the headline experiment
│   ├── run_dispatch_train.py
│   └── plot_results.py
├── docs/
│   └── CALIBRATION.md                # every behavioural rule, cited or flagged as assumption
├── tests/
│   ├── test_network.py
│   ├── test_demand.py
│   ├── test_vehicle.py
│   └── test_sim.py
├── notebooks/analysis.ipynb
└── runs/                            # logs, checkpoints (gitignored)
```

---

## 3. Synthetic network (`network.py`)

A corridor-based road network, not a full city map: `K` major corridors (matching
Lagos's well-documented radial structure — a handful of dominant arterial routes
into/out of Island and Mainland business districts, chronically congested at
predictable chokepoints like bridges) with junctions where routes cross. Model
as a graph: nodes = junctions/stops, edges = road segments with a
`base_travel_time` and a `congestion_sensitivity` (how much travel time
increases with vehicle density on that edge — bridges and CBD-adjacent segments
get a steeper curve, matching documented chronic-bottleneck locations).

```python
def generate_network(n_corridors, n_junctions_per_corridor, gen) -> Graph
def travel_time(edge, current_density) -> float   # BPR-style congestion function
```

---

## 4. Demand (`demand.py`)

Passenger arrivals generated per origin-destination pair via a time-varying
Poisson process, with a **bimodal peak structure** (morning inbound-to-CBD,
evening outbound) — the single most robust, well-documented fact about Lagos
commute patterns, and the one demand-side calibration point worth being
confident about. `arrival_rate(origin, dest, time_of_day)` scales with
distance-to-CBD and time-of-day peak factor.

---

## 5. Vehicle agents (`vehicle.py`)

Two agent types, both state machines:

**Danfo/keke (informal)**:
```
states: WAITING_AT_STOP (accumulating passengers) → DEPART (threshold met
        or max_wait exceeded) → EN_ROUTE (subject to congestion, §3) →
        DROP/PICKUP (probabilistic mid-route deviation toward higher local
        demand) → back to WAITING_AT_STOP at new position
```
`departure_threshold` (fraction of nominal capacity) and `max_wait_before_departure_anyway`
are the two key tunable parameters — sweep both in §9b.

**Formal bus (baseline, `formal_baseline.py`)**:
```
states: fixed headway departure regardless of load, fixed route, hard capacity
        cap (excess demand queues for the next scheduled vehicle)
```

---

## 6. Batched simulation loop (`sim.py`)

Run `B` independent simulated days (different demand-seed realisations)
simultaneously as tensor state, advancing all vehicles and all passenger queues
in lockstep time steps — same batching principle as the rest of the portfolio.
Per step: compute local passenger-count-per-stop, local vehicle-density-per-edge,
update travel times (§3), resolve threshold-departure decisions, move vehicles
along edges, resolve pickups/dropoffs. All as `[B, N]`-shaped tensor ops; avoid a
Python loop over individual vehicles or passengers.

```python
def step(sim_state, cfg, gen) -> sim_state
def run_day(cfg, gen, n_steps) -> metrics   # mean wait time, mean travel time,
                                              # vehicle utilisation, total trips served
```

---

## 7. Metrics

- **Mean passenger wait time** per corridor, per time-of-day.
- **Mean in-vehicle travel time** (affected by congestion + route deviation).
- **Vehicle utilisation** (fraction of capacity carried, on average).
- **Throughput** (total passenger-trips served per simulated day) at fixed fleet
  size — the fairest basis for the informal-vs-formal comparison (§9a), since
  giving one regime more vehicles would trivially favour it.

---

## 8. Learned dispatch policy (`dispatch_policy.py`)

A small policy (contextual bandit or lightweight RL) that an informal operator
uses to decide, at each stop, whether to depart now or wait longer, and whether
to deviate toward a higher-demand junction — replacing the fixed-threshold rule
in §5 with a learned one, trained to maximize the operator's trips-per-hour
(a reasonable proxy for informal-driver incentives, which are revenue-driven,
not headway-driven). This is a genuinely open question, not assumed: **does the
learned policy converge toward something that resembles scheduled behaviour
(regular departure intervals) once enough operators are simultaneously
optimizing, or does it stay irregular even under optimization pressure?** That's
the second, smaller contribution alongside §9a.

---

## 9. Experiments for the write-up

### 9a. Headline experiment — informal vs formal, same network, same demand

Run both vehicle types (§5) on the identical network (§3) and demand (§4), fleet
size held equal, across a full simulated day with the morning/evening peak
structure. Report mean wait time, travel time, and throughput for each regime,
broken out by time-of-day (off-peak vs peak) — the central hypothesis worth
testing rather than assuming: informal transit's flexibility (departure when
full, route deviation toward demand) should plausibly **outperform** fixed
scheduling specifically under peak, uneven demand, while underperforming (longer
average wait) under low, steady demand where a predictable schedule beats a
threshold rule. Report whichever result the simulation actually gives.

### 9b. Supporting experiments

1. **Threshold sensitivity**: sweep `departure_threshold` and
   `max_wait_before_departure_anyway` — show the wait-time/utilisation tradeoff
   curve informal operators are implicitly navigating.
2. **Congestion sensitivity**: how much does the informal regime's advantage
   (or disadvantage) from 9a shrink as bottleneck congestion (§3) is made more
   severe — does flexibility stop mattering once travel time dominates wait time?
3. **Learned-policy convergence** (§8): does the trained dispatch policy's
   emergent departure pattern become more regular (lower inter-departure
   variance) as more agents optimize simultaneously? Report inter-departure
   variance vs. population size.

---

## 10. Three-to-four-week plan

**Week 1 — Network and demand.** `network.py`, `demand.py`,
`docs/CALIBRATION.md` (flag every assumption honestly). Sanity-check the
bimodal peak demand pattern visually before building agents on top of it.

**Week 2 — Both vehicle types, batched sim.** `vehicle.py`,
`formal_baseline.py`, `sim.py`. Get one full simulated day running for both
regimes; verify metrics (§7) are sane (e.g. formal-bus wait time should be
roughly bounded by half the headway in steady-state, a known queueing-theory
check).

**Week 3 — The contribution.** `scripts/run_comparison.py` (§9a) across many
demand-seed realisations for statistical confidence; the peak/off-peak
breakdown. This is the week not to compress.

**Week 4 — Learned policy, supporting experiments, write-up.**
`dispatch_policy.py` (§8), §9b's three experiments, write the report centred on
§9a, populate README, tag `v1.0`.

Git rule: push on day one; push every session; tag each week's milestone.

---

## 11. Config (single source of truth)

```python
@dataclass
class Config:
    # network
    n_corridors: int = 6
    n_junctions_per_corridor: int = 8
    bottleneck_edges_frac: float = 0.15
    # demand
    peak_multiplier: float = 3.0
    morning_peak_hour: float = 7.5
    evening_peak_hour: float = 17.5
    # informal vehicle
    departure_threshold: float = 0.75
    max_wait_before_departure_anyway: float = 8.0   # minutes
    deviation_prob: float = 0.15
    soft_capacity_overload: float = 1.2
    # formal vehicle
    headway_minutes: float = 15.0
    hard_capacity: int = 30
    # sim
    n_days: int = 200          # parallel batch
    sim_hours: float = 16.0
    step_minutes: float = 1.0
    fleet_size: int = 40
    # dispatch policy
    policy_lr: float = 1e-3
    policy_steps: int = 5000
    device: str = "cuda"
    seed: int = 0
```

---

## 12. Risks and their fixes

| Risk | Symptom | Fix |
|---|---|---|
| Informal always wins or always loses trivially | 9a shows no time-of-day-dependent crossover | check demand isn't uniformly peaked or uniformly flat — the crossover needs real peak/off-peak contrast; verify `max_wait_before_departure_anyway` isn't so low it makes the threshold rule irrelevant |
| Formal-bus wait time looks wrong | doesn't match queueing-theory sanity check | verify headway-based departure logic; check hard-capacity queueing isn't silently dropping passengers instead of carrying them to the next bus |
| Calibration feels thin | can't cite most parameters confidently | this is expected per §0 — label assumptions as assumptions in `CALIBRATION.md` and lean on the *comparison* (informal vs formal on identical assumptions) rather than absolute numbers, since the comparison is robust to some calibration uncertainty even when point estimates aren't |
| GPU underused | simulation runs are small (few thousand agents) | batch across many parallel simulated days (`n_days`) rather than trying to make a single day's simulation bigger — that's where this project's scale actually is |
| Dispatch policy collapses to trivial always-depart-immediately | policy under-values passenger throughput | check the reward function actually penalizes low-occupancy departures, not just rewards trip count |

---

## 13. Write-up outline (4–6 pages)

1. Why informal transit is structurally different from the fixed-route model
   almost all transit simulation assumes (§1), and why Lagos is a distinctive
   case study — with an explicit statement of what's assumption vs. documented
   fact (§0).
2. Method: network, demand, both vehicle models, batched simulation.
3. **The contribution**: §9a's informal-vs-formal comparison, broken out by
   time-of-day, and its interpretation.
4. Supporting results: §9b, and the dispatch-policy convergence question (§8).
5. Limitations: single-city calibration with several assumed (not measured)
   behavioural parameters, no real trip-time validation data, DC-style
   simplified congestion model.
6. Reproducibility: one command, pinned seeds, hardware and wall-clock reported.
