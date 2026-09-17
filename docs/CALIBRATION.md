# Calibration

This document grounds every behavioural rule in DanfoSim in either a cited
source describing how Lagos's informal transit system operates, or flags it
explicitly as a reasonable modelling assumption where no such source exists.
Per the architecture's honesty principle (§0): **there is no substantial
existing benchmark of informal-vs-formal transit simulated head-to-head on
the same network**, so this project's calibration is more exposed to
challenge than a project with a mature reference model to compare against.
Read every "documented pattern" claim below as "the qualitative direction is
well attested"; read every numeric parameter value as a starting point for
the sensitivity sweeps in §9b, not a validated point estimate.

## Documented patterns (reasonably confident)

| Rule | Where it's modelled | Basis |
|---|---|---|
| Radial corridor structure: a handful of dominant arterial routes into/out of Island and Mainland business districts | `network.py`: K corridors sharing one CBD hub node | Widely documented feature of Lagos's road network and transit geography — travel is structured around a small number of bridges/expressways connecting the mainland to Lagos Island, not a dense uniform grid. |
| Chronic bottlenecks at bridges / CBD-adjacent segments | `network.py`: the edges closest to the CBD hub in each corridor are flagged `is_bottleneck` with a steeper BPR congestion curve (`congestion_sensitivity`, `capacity`) | Well-documented: the mainland–island bridges (e.g. Third Mainland Bridge corridor) are Lagos's most consistently cited chokepoints. |
| Bimodal commute demand: morning peak inbound to CBD, evening peak outbound | `demand.py`: `arrival_rate` peaks inbound stops at `morning_peak_hour`, outbound (CBD) stops at `evening_peak_hour` | The single most robust, well-documented fact about Lagos commute patterns — this is the one demand-side calibration point the architecture explicitly flags as worth high confidence. |
| No fixed schedule; a danfo/keke departs when sufficiently full | `vehicle.py`: `departure_threshold` fraction-of-capacity rule | Widely and consistently reported as *the* defining operational difference between Lagos's informal minibus/tricycle system and scheduled transit. |
| Soft, elastic capacity — a danfo can exceed its nominal seat count under pressure | `vehicle.py`: `soft_capacity_overload` allows boarding above `nominal_capacity` | Commonly reported (extra standing/seated passengers beyond nominal capacity), though the specific multiplier (1.2×) is this project's assumption, not a measured figure. |
| Informal, uncentralised fare/route flexibility, including mid-route deviation toward higher demand and "dropping" passengers short of the nominal terminus | `vehicle.py`: `deviation_prob`-gated DROP mechanic | The qualitative behaviour (drivers route-flex toward revenue-maximising demand, short-turn routes) is a commonly described feature of Lagos danfo operation; the specific 15% per-arrival probability is **an assumption**, not a cited rate — swept in §9b. |
| Formal transit (BRT-style) uses fixed routes, fixed headway, and hard capacity limits | `formal_baseline.py` | Standard, well-documented operating model for scheduled fixed-route transit generally (not Lagos-specific) — used here purely as the comparison baseline, not as a claim about any specific existing Lagos BRT line's exact parameters. |

## Assumptions flagged explicitly (no strong citation; reasonable but not validated)

- **`nominal_capacity = 14`** — typical of a Lagos danfo minibus's nominal
  seating; treated as a round-number assumption, not a measured fleet
  average. Real vehicles vary (keke napep tricycles seat far fewer).
- **`base_arrival_rate = 0.35` passengers/minute at an average off-peak
  stop** and the overall demand magnitude — chosen (via a manual sweep, see
  `docs/CALIBRATION.md` git history / project memory) to produce single-digit
  to ~30-minute wait times under peak load, a plausible order of magnitude
  for informal minibus/tricycle waits, rather than derived from a ridership
  count. Earlier, higher settings (0.8) produced 100+ minute mean waits under
  peak — clearly implausible — which is itself informative: the model is
  sensitive to this assumption, so §9b's sensitivity sweeps matter more than
  any single point estimate. The project leans on the *comparison* (§9a:
  informal vs. formal under identical demand) being robust to this
  uncertainty, not on the absolute wait-time numbers.
- **`peak_multiplier = 3.0`, `peak_width_hours = 1.2`** — the peak-to-off-peak
  demand ratio and peak sharpness are assumed; the *existence* of AM/PM peaks
  is well documented, their exact magnitude is not calibrated against a
  ridership survey.
- **`departure_threshold = 0.75`, `max_wait_before_departure_anyway = 8 min`**
  — plausible values for a load-threshold departure rule; swept across a
  range in §9b rather than treated as fixed truths.
- **`deviation_prob = 0.15`** — see above; swept in §9b.
- **Distance-from-CBD demand gradient** (`demand.py`: outer stops scale
  `0.5 + 0.5 * hops/(J-1)` relative to CBD-adjacent stops) — a monotonic
  "more suburban population commutes in" assumption, not derived from a
  population density map.
- **Fixed travel time per edge, set once at trip/leg departure from that
  edge's congestion level at that instant** (not recomputed continuously
  while the vehicle is mid-edge) — a standard discrete-time simplification
  of BPR-style congestion models, not a claim that real travel time is
  static once a trip begins.
- **A "dropped" passenger's trip is counted as completed** when an informal
  vehicle short-turns (§ vehicle.py DROP mechanic) — a simplification. In
  reality a dropped passenger may need a second vehicle to finish their
  journey (an informal transfer), which this model does not represent; this
  likely *overstates* informal throughput and *understates* informal
  effective travel time relative to a model that tracked transfers.
- **Formal fixed-route buses dwell briefly (`dwell_minutes = 0.5`) at every
  intermediate junction to board passengers, while informal vehicles perform
  a zero-dwell rolling curbside pickup at junctions they pass without
  deviating** — both are modelling necessities (without *some* intermediate
  boarding mechanism, demand at non-terminus stops would never be served),
  not measured dwell times. The asymmetry (formal pays a dwell-time cost,
  informal doesn't) is a deliberate, defensible reflection of danfo/keke's
  documented curbside/hand-signal boarding flexibility versus a scheduled
  bus's designated-stop model — but it is a modelling choice that favours
  informal's throughput, and should be read as such when interpreting §9a.
- **Formal headway-governed departure applies only at each route's terminus**
  (where a bus starts a fresh leg after reversing direction); intermediate
  stops use a fixed, load-independent dwell instead of the headway. This is
  the standard interpretation of "fixed headway" (a published departure
  frequency from the terminus) rather than a claim that intermediate
  boarding times are precisely 0.5 minutes.
- **Equal fleet size across regimes (§9a, §7)** is a deliberate experimental
  control (per the architecture: giving one regime more vehicles would
  trivially favour it), not a claim that Lagos's actual informal and formal
  fleets are equally sized.
- **CPU-only PyTorch was used for development** on this machine; the target
  hardware (Legion 7 Pro, 16GB GPU) should use a CUDA build for the full
  `n_days`-batched runs described in §9a/§9b — this affects wall-clock time,
  not simulation semantics.

## How to challenge or improve this calibration

Every assumption above is a named `Config` field (see `danfosim/config.py`)
and every one used in §9b's sensitivity sweeps should be read as "here is how
much the headline result depends on this specific unvalidated number" —
that dependence, not the point estimate itself, is the honest deliverable
when no ground-truth Lagos trip-level dataset is available.
