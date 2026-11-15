"""Conjunction screening: find upcoming close approaches between tracked objects.

Pipeline
  1. Altitude-shell prefilter   drop objects whose altitude band overlaps nobody
  2. Coarse sweep               propagate everything on a time grid, KD-tree per step
  3. Linear TCA estimate        closest-approach time from relative pos/velocity
  4. Dedupe                     one row per encounter
  5. SGP4 refinement            true minimum distance via bounded 1-D optimisation
  6. Risk score                 heuristic collision probability (see caveats)

CAVEATS:
  * TLEs carry no covariance. Position uncertainty is *assumed* to grow with TLE
    age (SIGMA_BASE_KM + SIGMA_GROWTH_KM_PER_DAY * age). The resulting Pc is a
    screening heuristic, NOT an operational probability.
  * Miss distances from TLEs are only good to ~1-3 km (more for old elements).

CLI (for a cron job):
    python conjunctions.py --groups active fengyun-1c-debris --hours 24 \
        --threshold 10 --out conjunctions.parquet
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sgp4.api import SatrecArray
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree

import orbits
from orbits import KIND_DEBRIS, MU, R_EARTH, Catalog, to_jd

V_REL_MAX = 16.0            # km/s, upper bound on relative speed in Earth orbit
SIGMA_BASE_KM = 0.5         # assumed 1-sigma position error at TLE epoch
SIGMA_GROWTH_KM_PER_DAY = 1.0
DEFAULT_HBR_KM = 0.010      # combined hard-body radius (10 m) when sizes unknown


# ------------------------------------------------------------------ 1. prefilter
def altitude_bands(cat: Catalog) -> tuple[np.ndarray, np.ndarray]:
    """Perigee and apogee altitude (km) for every object, from mean elements."""
    n = np.array([s.no_kozai for s in cat.sats]) / 60.0          # rad/s
    e = np.array([s.ecco for s in cat.sats])
    a = (MU / n**2) ** (1 / 3)
    return a * (1 - e) - R_EARTH, a * (1 + e) - R_EARTH


def shell_prefilter(perigee, apogee, margin_km: float, block: int = 500) -> np.ndarray:
    """True for objects whose [perigee, apogee] band overlaps another object's.

    Two objects whose altitude bands never overlap (+margin) cannot collide.
    """
    n = len(perigee)
    keep = np.zeros(n, dtype=bool)
    for s in range(0, n, block):
        p = perigee[s:s + block, None]
        a = apogee[s:s + block, None]
        ov = (perigee[None, :] <= a + margin_km) & (apogee[None, :] >= p - margin_km)
        ov[np.arange(ov.shape[0]), np.arange(s, s + ov.shape[0])] = False  # ignore self
        keep[s:s + block] = ov.any(axis=1)
    return keep


# ------------------------------------------------------------------ 5. refinement
def refine_pair(sat_a, sat_b, jd0: float, fr0: float, tca_s: float, window_s: float):
    """Exact (SGP4) closest approach of two satellites near `tca_s` seconds after start.

    Returns (tca_s, miss_km, rel_speed_km_s).
    """
    def dist(s: float) -> float:
        fr = fr0 + s / 86400.0
        ea, ra, _ = sat_a.sgp4(jd0, fr)
        eb, rb, _ = sat_b.sgp4(jd0, fr)
        if ea or eb:
            return 1e9
        return float(np.linalg.norm(np.subtract(ra, rb)))

    res = minimize_scalar(dist, bounds=(tca_s - window_s, tca_s + window_s),
                          method="bounded", options={"xatol": 1e-3})
    s = float(res.x)
    fr = fr0 + s / 86400.0
    _, _, va = sat_a.sgp4(jd0, fr)
    _, _, vb = sat_b.sgp4(jd0, fr)
    return s, float(res.fun), float(np.linalg.norm(np.subtract(va, vb)))


# ------------------------------------------------------------------ 6. risk
def sigma_from_age(age_days) -> np.ndarray:
    return SIGMA_BASE_KM + SIGMA_GROWTH_KM_PER_DAY * np.abs(age_days)


def collision_probability(miss_km, sigma_km, hbr_km: float = DEFAULT_HBR_KM):
    """Small-target approximation of Pc for an isotropic Gaussian encounter.

    Pc ~= HBR^2 / (2 sigma^2) * exp(-miss^2 / (2 sigma^2)),  valid for HBR << sigma.
    Real systems use full 3-D covariances in the encounter plane; we don't have them.
    """
    s2 = np.asarray(sigma_km) ** 2
    pc = (hbr_km**2 / (2 * s2)) * np.exp(-np.asarray(miss_km) ** 2 / (2 * s2))
    return np.clip(pc, 0.0, 1.0)


def risk_level(pc: float) -> str:
    if pc >= 1e-4:      # common operator manoeuvre-planning threshold
        return "HIGH"
    if pc >= 1e-6:
        return "WATCH"
    return "LOW"


# ------------------------------------------------------------------ main entry
def screen(
    cat: Catalog,
    start: datetime,
    hours: float = 24.0,
    step_s: float = 10.0,
    threshold_km: float = 10.0,
    min_rel_speed_kms: float = 0.05,
    include_debris_pairs: bool = False,
    chunk: int = 60,
    progress: Optional[Callable[[float], None]] = None,
) -> pd.DataFrame:
    """Screen the whole catalog for approaches closer than `threshold_km`.

    step_s trades speed for safety: the KD-tree radius is widened by
    V_REL_MAX*step/2 so no encounter can slip between two time samples.
    min_rel_speed_kms drops docked / co-orbiting pairs (they are not conjunctions).
    """
    start = start.astimezone(timezone.utc) if start.tzinfo else start.replace(tzinfo=timezone.utc)
    jd0, fr0 = to_jd(start)

    # --- 1. prefilter ---
    _, _, ok0 = orbits.propagate(cat, start)
    perigee, apogee = altitude_bands(cat)
    margin = threshold_km * 1.5
    sel = np.flatnonzero(ok0 & shell_prefilter(perigee, apogee, margin))
    if len(sel) < 2:
        return pd.DataFrame()
    sats = [cat.sats[i] for i in sel]
    arr = SatrecArray(sats)
    kinds = cat.kinds[sel]

    # --- 2/3. coarse sweep + linear TCA ---
    coarse = margin + V_REL_MAX * step_s / 2
    steps = np.arange(0.0, hours * 3600.0 + step_s, step_s)
    hits = []   # rows of (i, j, tca_s, miss_lin)
    for c0 in range(0, len(steps), chunk):
        secs = steps[c0:c0 + chunk]
        e, r, v = arr.sgp4(np.full(len(secs), jd0), fr0 + secs / 86400.0)
        for t in range(len(secs)):
            ids = np.flatnonzero(e[:, t] == 0)
            if len(ids) < 2:
                continue
            pr = cKDTree(r[ids, t, :]).query_pairs(coarse, output_type="ndarray")
            if not len(pr):
                continue
            i, j = ids[pr[:, 0]], ids[pr[:, 1]]
            if not include_debris_pairs:
                m = ~((kinds[i] == KIND_DEBRIS) & (kinds[j] == KIND_DEBRIS))
                i, j = i[m], j[m]
            if not len(i):
                continue
            dr = r[j, t, :] - r[i, t, :]
            dv = v[j, t, :] - v[i, t, :]
            vv = (dv * dv).sum(axis=1)
            good = vv > min_rel_speed_kms**2
            tau = -(dr * dv).sum(axis=1) / np.where(good, vv, 1.0)
            good &= np.abs(tau) <= step_s / 2            # closest approach inside this bin
            miss = np.linalg.norm(dr + dv * tau[:, None], axis=1)
            good &= miss <= margin
            if good.any():
                hits.append(np.column_stack([i[good], j[good],
                                             secs[t] + tau[good], miss[good]]))
        if progress:
            progress(min(1.0, (c0 + chunk) / len(steps)))

    if not hits:
        return pd.DataFrame()

    # --- 4. dedupe: one row per encounter ---
    df = pd.DataFrame(np.vstack(hits), columns=["i", "j", "tca_s", "miss_lin"])
    df[["i", "j"]] = df[["i", "j"]].astype(int)
    df = df.sort_values(["i", "j", "tca_s"]).reset_index(drop=True)
    new = (df.i.ne(df.i.shift()) | df.j.ne(df.j.shift()) | (df.tca_s.diff() > 3 * step_s))
    df["g"] = new.cumsum()
    df = df.loc[df.groupby("g").miss_lin.idxmin()]

    # --- 5. exact refinement ---
    rows = []
    for rw in df.itertuples():
        tca_s, miss, vrel = refine_pair(sats[rw.i], sats[rw.j], jd0, fr0, rw.tca_s, step_s)
        if miss > threshold_km:
            continue
        ca, cb = int(sel[rw.i]), int(sel[rw.j])
        tca_jd = jd0 + fr0 + tca_s / 86400.0
        age_a = tca_jd - (cat.sats[ca].jdsatepoch + cat.sats[ca].jdsatepochF)
        age_b = tca_jd - (cat.sats[cb].jdsatepoch + cat.sats[cb].jdsatepochF)
        rows.append(dict(
            idx_a=ca, idx_b=cb,
            name_a=cat.names[ca], norad_a=int(cat.norad[ca]), kind_a=orbits.KIND_LABELS[int(cat.kinds[ca])],
            name_b=cat.names[cb], norad_b=int(cat.norad[cb]), kind_b=orbits.KIND_LABELS[int(cat.kinds[cb])],
            tca_utc=start + timedelta(seconds=tca_s),
            miss_km=miss, rel_speed_kms=vrel, tle_age_a_d=age_a, tle_age_b_d=age_b,
        ))
    if not rows:
        return pd.DataFrame()

    # --- 6. risk ---
    out = pd.DataFrame(rows)
    sigma = np.sqrt(sigma_from_age(out.tle_age_a_d) ** 2 + sigma_from_age(out.tle_age_b_d) ** 2)
    out["sigma_km"] = sigma
    out["pc"] = collision_probability(out.miss_km.values, sigma)
    out["risk"] = out.pc.map(risk_level)
    return out.sort_values("miss_km").reset_index(drop=True)


# ------------------------------------------------------------------ CLI
def main() -> None:
    ap = argparse.ArgumentParser(description="Screen CelesTrak groups for conjunctions")
    ap.add_argument("--groups", nargs="+", default=["active"], help="CelesTrak GROUP names")
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--threshold", type=float, default=10, help="miss distance km")
    ap.add_argument("--step", type=float, default=10, help="coarse time step, seconds")
    ap.add_argument("--debris-pairs", action="store_true", help="include debris-debris pairs")
    ap.add_argument("--out", default="conjunctions.parquet")
    a = ap.parse_args()

    tles = []
    for g in a.groups:
        tles += orbits.fetch_tles(g)
    cat = orbits.build_catalog(tles)
    print(f"{len(cat.names)} objects loaded")
    df = screen(cat, datetime.now(timezone.utc), a.hours, a.step, a.threshold,
                include_debris_pairs=a.debris_pairs,
                progress=lambda p: print(f"\r{p:5.0%}", end="", flush=True))
    print(f"\n{len(df)} conjunctions found")
    if a.out.endswith(".parquet"):
        df.to_parquet(a.out, index=False)
    else:
        df.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()