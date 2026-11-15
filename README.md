# 🛰️ Orbital Collision Radar

An interactive Streamlit app that shows **live satellites and space junk**, finds **close calls** between
them, and simulates **what a collision or explosion would do**.

**Why it matters:** thousands of satellites and debris pieces share Earth's orbit at ~7 km/s. In 2009 two
satellites collided and created thousands of new fragments, which made the next collision more likely.
**Questions the app answers:** *Which objects will pass dangerously close soon? What would a collision do to everything else?*

## What it does
| Tab | What you can do |
|---|---|
| 🌍 Explore | Live 3D globe of up to 6,000 objects. Search by name, hover for names, click for details and the next-lap path, play the time-lapse. |
| ⚠️ Close calls | Screens the catalog for pairs passing within a chosen distance, ranks them, and replays each one on the globe. |
| 💥 What if? | Simulates a collision, an explosion, or the real 2009 Iridium-Cosmos crash. Shows the debris cloud, how it spreads, and which satellites are in danger. |

## Data
- **Source:** [CelesTrak](https://celestrak.org) (Dr T.S. Kelso), republishing the US Space Force public catalog. Free, no key.
- **Records:** orbital elements (CSV/OMM format), updated every 2 hours. One row per object.
- **Cleaning:** duplicates removed, decayed/invalid objects dropped, 6-digit catalog numbers handled (CelesTrak stopped serving
  them in the TLE format on 2026-07-11, so the app reads CSV and re-encodes them as Alpha-5 TLE lines), 9-digit analyst
  objects skipped. A **data-quality report** is shown inside the app.
- **Polite access:** cached for 2 hours, stops asking after a server error (per CelesTrak's usage policy), and falls back to a
  bundled snapshot in `data/` if CelesTrak is unavailable.
- **Accuracy:** about 1–3 km, worse as a record ages. This is a learning/screening tool, **not** an official warning system.

## Method
1. **SGP4** orbit model → position and velocity of every object at any time.
2. **Close-call screening:** altitude-shell prefilter → KD-tree search every 10 s → linear closest-approach estimate →
   exact refinement with SGP4. Validated against a brute-force search (60/60 encounters found).
3. **Risk score:** heuristic probability from the gap, the age of the data and its assumed uncertainty.
4. **Breakup simulator:** simplified NASA Standard Breakup Model (fragment count, size, velocity), propagated with two-body + J2.
5. **Exposure ranking:** local fragment density around each satellite over time.

## Validation
- Orbit-drift from Earth's bulge: numeric −5.000°/day vs analytic −5.002°/day.
- Fragment propagator vs SGP4: 0.3 km after 6 hours.
- 2009 replay: relative speed 11.55 km/s (real ≈ 11.65); ~1,200 predicted fragments vs 2,000+ tracked.

## Limitations
Simplified breakup model; no air drag or solar pressure; object masses are assumptions; TLE-level accuracy.

## Run it
```
python -m pip install -r requirements.txt
python snapshot.py        # optional: saves a fallback copy of the data into data/
streamlit run app.py
```