# Does going informal actually help? Simulating Lagos danfo transit against a fixed-route baseline

Lagos gets around mostly on danfo minibuses and keke napep tricycles — vehicles that don't run on a timetable, don't have fixed stops in the way a bus route does, and are driven by people who are trying to make money, not hit a schedule. A danfo leaves when it's full enough, not at a scheduled time. Drivers will cut a route short if a better fare is waiting somewhere else. None of this is how transit simulation normally works — almost every agent-based transit model out there assumes fixed routes and fixed headways, because that's how the transit systems most researchers study actually operate.

So here's the question I wanted an actual number for, not just an intuition: is that flexibility worth anything? If you gave Lagos a fixed-route, fixed-headway system instead — same roads, same number of vehicles, same passengers wanting to go the same places — would it be worse, better, or does it depend on when you look?

I built a simulation to find out. This is what I found, and just as importantly, what I don't think I've earned the right to claim yet.

## Nobody has actually run this comparison

I looked, and there isn't a real head-to-head benchmark of informal versus formal transit simulated on the same network with the same demand. Fixed-route transit simulation is a mature field — MATSim and its relatives have been doing large-scale microsimulation for decades. Research on informal transit in African cities exists too, but it's mostly qualitative or policy-focused: describing how danfo and matatu systems behave, not building a computational model you could run a controlled experiment on.

That's good news and bad news. Good, because it means the comparison itself is a real contribution rather than a rehash. Bad, because there's very little to calibrate against. I don't have ridership counts, GPS traces, or fare data for Lagos's informal network. What I have is a set of well-documented *qualitative* facts about how the system operates — no fixed schedule, threshold-based departure, route flexibility, a strong morning/evening commute peak — and I built the simulation's structure around those, while being upfront about which numbers are backed by something and which ones I picked because they produced sane behavior. Every parameter in the model is logged in `docs/CALIBRATION.md` along with which category it falls into. I'd rather someone read that file and disagree with a specific number than take the headline result at face value.

## What the model actually simulates

The network is six radial corridors feeding into one shared CBD node, which is the honest shape of Lagos's road geography — a handful of arterial routes converging on Lagos Island and the Mainland business districts, with the well-known chokepoints (the bridges, basically) sitting right at the CBD end of each corridor. Those edges get a steeper congestion curve than the rest of the network.

Passenger demand is a time-varying Poisson process with a morning peak (people heading to the CBD) and an evening peak (people heading home) — the one demand fact about Lagos I feel genuinely confident calibrating, since it's about as well-documented as commute patterns get anywhere.

The informal vehicles are a state machine: wait and pick up passengers until you hit a load threshold or a max-wait timer, then go; while en route, there's a small chance of cutting the trip short at whichever junction you're passing if it looks like a better move (the "drop and restart somewhere better" behavior that's actually pretty central to how danfo drivers operate); and while passing intermediate junctions without cutting the trip short, pick up anyone waiting there too, because that's how curbside hailing actually works — you don't need to formally "stop" to take on a passenger.

The formal baseline is what you'd expect: fixed route, fixed headway from the terminus, a hard capacity limit, and — this took a bug to discover — it also needs to actually stop and board passengers at every junction along the route, not just the terminus. My first version only boarded at the start of the line, which meant every intermediate stop's demand just piled up forever with nobody ever there to serve it. Once I added a short dwell stop at each junction, the formal baseline started behaving like an actual bus route instead of an express service that skips its own stops.

Everything runs as batched tensor operations — 200 simulated days at once, no Python loop over individual vehicles or passengers — so a full day's simulation for both regimes takes a few seconds on a CPU. I never actually needed the GPU this project was scoped around; the fleet sizes involved (tens of vehicles per corridor) are just too small to be compute-bound. That's a useful thing to know for anyone repeating this: don't over-invest in GPU infrastructure for a simulation at this scale.

## The main result

Same network, same demand, same fleet size (40 vehicles) for both regimes, run across 200 simulated days so I could put a confidence interval on the numbers instead of reporting a single lucky run.

| | mean wait | mean travel time | utilisation | trips served |
|---|---|---|---|---|
| Informal | 49.8 min (95% CI 49.3–50.3) | 19.8 min | 0.54 | 297,407 |
| Formal | 72.4 min (95% CI 72.1–72.7) | 37.8 min | 0.27 | 129,400 |

![Informal vs formal comparison, broken out by time of day](figures/comparison.png)

Informal wins, and it's not close. Lower wait time, lower travel time, double the utilisation, more than double the trips served with the identical fleet. Breaking it down by time of day is where it gets more interesting than "informal wins":

- **Morning peak**: informal wait is 88.3 min against formal's 97.0 min — informal still wins, but the gap is proportionally smaller than off-peak. Travel time diverges hard here too (35.8 vs 55.1 min), because the formal fleet is stuck doing the full fixed route through worsening congestion while informal vehicles are short-turning out of it.
- **Off-peak**: informal's advantage is largest here in relative terms — 21.7 min against 39.7 min. This surprised me a little; I'd expected the threshold-departure rule to actually hurt informal under low, steady demand, since a danfo waiting to fill up when passengers trickle in slowly should lose to a bus that leaves on a predictable schedule regardless of load. That crossover didn't show up. The `max_wait_before_departure_anyway` safety valve (capped at 8 minutes by default) is apparently doing enough work to prevent the threshold rule from ever really costing informal much, even off-peak.
- **Evening peak**: this is the one bucket where the two regimes are statistically indistinguishable — 20.6 min for informal versus 20.2 min for formal, and their 95% confidence intervals overlap. I want to be honest about this rather than bury it: the "informal wins everywhere" headline has one real exception, and I don't have a fully satisfying explanation for why evening peak specifically is where they converge. My best guess is that outbound demand from the CBD is more spatially concentrated (everyone starts from the same hub node) than inbound demand is, which plays more to a scheduled system's strengths, but I haven't verified that mechanism directly.

So the honest one-line version isn't "informal transit is strictly better." It's "informal transit's flexibility pays off almost everywhere in this model, and the one place it doesn't is a real result, not noise, and worth investigating further rather than smoothing over."

## What actually moves the numbers

Two follow-up sweeps, both run informal-only or as an informal/formal pair, again at the full 200-day batch.

**Threshold sensitivity.** I swept `departure_threshold` (how full a vehicle needs to be before it's allowed to leave) from 0.5 to 1.0, and `max_wait_before_departure_anyway` from 4 to 15 minutes. The threshold itself barely matters — wait time moves by less than a minute across the whole 0.5–1.0 range at a fixed max-wait. What actually drives the tradeoff is the max-wait cap: pushing it from 4 to 15 minutes takes mean wait from 48.8 to 61.0 minutes but takes utilisation from 0.47 to 0.65. That's the real dial an informal operator (or a policymaker trying to nudge their behavior) is turning — not "how full is full enough," but "how long are you willing to let people wait before you leave anyway."

**Congestion sensitivity.** I scaled the background traffic congestion term from a quarter of its default severity up to four times it, running both regimes at each level. Informal's absolute wait-time advantage grows a lot as congestion gets worse — from a 16-minute edge at the lightest congestion setting to a 330-minute edge at the heaviest, and it's statistically significant at every single level I tested. But the *relative* advantage — informal's wait as a fraction of formal's — stays close to constant, around 1.4 to 1.5x, across the whole range. In other words: flexibility doesn't stop mattering once congestion gets bad enough for travel time to dominate wait time. If anything the raw benefit gets bigger, it just doesn't get *proportionally* bigger.

One thing worth flagging about the congestion model itself: I initially built it so that only the simulated transit vehicles counted toward road congestion, and it turned out a fleet of 40 vehicles spread across a few dozen road segments is nowhere near dense enough to ever load a bottleneck edge up to a congesting level on its own. The sweep was completely flat until I fixed this — real Lagos bridge congestion comes overwhelmingly from general traffic, not danfo density, so I added a background traffic term tied to time of day and bottleneck status. That's disclosed as an assumption in the calibration doc, not something I'm pretending is measured.

## Does competition make dispatch more disciplined?

The last piece asks a genuinely different question: if you replace the fixed threshold rule with a policy that *learns* when to depart — trained to maximize something like revenue per trip while paying a cost for wait time, so it can't just collapse to "leave immediately, always" — does the emergent departure pattern get more regular as more drivers are competing for the same passengers? That's a real open question about informal transit systems: does market pressure eventually produce something that looks like a schedule, even without anyone imposing one?

In this model, no — the opposite happens. I trained a separate policy from scratch at each fleet size (10, 20, 40, 80 vehicles) so each one reaches its own equilibrium rather than just testing one policy out of its training conditions, and the variance in time between departures climbs sharply as the fleet grows:

| fleet size | inter-departure variance | coefficient of variation |
|---|---|---|
| 10 | 0.024 | 0.60 |
| 20 | 0.032 | 0.65 |
| 40 | 0.113 | 0.98 |
| 80 | 2.81 | 1.19 |

I didn't trust this number the first time I saw it — a jump like that at fleet_size=80 looked like it could easily be an artifact of pooling gaps across corridors that just happen to run at different average paces. So I decomposed it: within-corridor variance versus between-corridor variance. Between-corridor variance stays two orders of magnitude smaller than within-corridor at every single fleet size, and the coefficient of variation (which is scale-free, so it's not just "gaps got bigger") climbs the whole way too. That rules out the pooling artifact. The effect is real: individual operators get more erratic, not less, as more of them compete for the same stops.

My read on why: with more vehicles chasing the same demand pool, each individual vehicle's time-to-fill becomes more dependent on random luck about who else happens to be waiting at the same junction at the same time — competition adds noise to each operator's own cycle rather than averaging it out. That's a plausible real-world story too, and it's consistent with something people who've written about matatu and danfo systems have observed anecdotally: intense competition for passengers doesn't spontaneously produce anything resembling a schedule.

## Where I'd push back on this if I were reviewing it

A few things I don't think this model has earned the right to claim:

The demand numbers are tuned, not measured. I picked a base arrival rate that produces plausible-looking wait times (single digits to a few tens of minutes) rather than one backed by a ridership count, because my first attempt at a demand rate produced 100+ minute average waits that were obviously wrong. That's a real limitation, and it means I'd trust the *comparison* between regimes — which holds under the same demand assumptions for both — a lot more than I'd trust any of the absolute minute values on their own.

A dropped passenger, in this model, counts as a completed trip. When an informal vehicle short-turns and ends a trip early, the model doesn't track that the passenger might now need a second vehicle to actually finish their journey. That almost certainly overstates informal throughput and understates informal travel time relative to a model that handled transfers properly.

The formal and informal fleets aren't boarding symmetrically. Formal buses pay a fixed dwell cost at every stop; informal vehicles get a free rolling pickup with no dwell time at all, reflecting how curbside hailing actually works. That's a defensible modeling choice, not an oversight, but it's also a choice that mechanically favors informal, and I want that on the record rather than discovered by someone else later.

And single-city, single-network calibration means none of this should be read as "here's what Lagos wait times actually are." It should be read as "here's what happens to the *comparison* between two operating models when you hold demand and road network fixed" — which is a narrower and more defensible claim.

## Running it

```
pip install -r requirements-lock.txt
pytest -q                                    # 35 tests, ~2 seconds
python scripts/run_comparison.py             # the headline result above
python scripts/run_sensitivity.py threshold
python scripts/run_sensitivity.py congestion
python scripts/run_dispatch_train.py         # the fleet-size sweep, ~90s on CPU
```

Everything above ran on CPU in well under a couple of minutes total, seeded and reproducible. `docs/CALIBRATION.md` has every parameter and whether it's cited or assumed; the code itself is the actual specification of the model, and the tests are the actual specification of what I believe it gets right.
