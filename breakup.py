"""Breakup simulator: fragment cloud generation, J2 propagation, exposure ranking.

MODEL (simplified NASA Standard Breakup Model; say "simplified" in your report):
  * fragment count   N(>=Lc) = 0.1 * M^0.75 * Lc^-1.71   (collision)
                     N(>=Lc) = 6 * Lc^-1.6               (explosion)
    M = total mass if catastrophic (impact energy / target mass >= 40 J/g),
    otherwise projectile_mass * v_rel^2 (v in km/s).
  * velocities       each fragment inherits one parent's velocity (mass-weighted) + its kick
  * fragment sizes   power law (Pareto) above the minimum size
  * area-to-mass     lognormal, centred on a size-dependent value (simplified)
  * delta-v          log10(dv[m/s]) ~ N(0.9*chi + 2.9, 0.4) collision, N(0.2*chi + 1.85, 0.4) explosion
                     chi = log10(A/M);   direction isotropic
  * propagation      two-body + J2 (RK4). NO drag, NO solar pressure, NO fragment-fragment effects.
Masses of real objects are not in TLEs: they are assumptions you can change.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sgp4.api import SatrecArray

import conjunctions as cj
import orbits
from orbits import KIND_DEBRIS, KIND_PAYLOAD, KIND_ROCKET, MU, R_EARTH, Catalog, to_jd

J2 = 1.08262668e-3
DEFAULT_MASS_KG = {KIND_PAYLOAD: 500.0, KIND_ROCKET: 1500.0, KIND_DEBRIS: 20.0}


# ------------------------------------------------------------------ cloud generation
@dataclass
class Cloud:
    r0: np.ndarray            # (N,3) km, TEME, at the event
    v0: np.ndarray            # (N,3) km/s
    lc: np.ndarray            # (N,) characteristic length, m
    n_estimated: int          # model's total fragment count >= lc_min (before capping)
    mass_eff_kg: float
    catastrophic: bool | None
    rel_speed_kms: float | None
    scenario: str


def _unit_vectors(rng, n):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _sizes_and_dv(rng, n, lc_min, mode):
    lc = np.minimum(lc_min * rng.random(n) ** (-1 / 1.71), 1.5)
    chi = np.clip(rng.normal(-0.8 - 0.5 * np.log10(lc / 0.1), 0.4), -2.0, 0.8)   # log10(A/M)
    mu = 0.9 * chi + 2.9 if mode == "collision" else 0.2 * chi + 1.85
    dv_ms = np.minimum(10 ** rng.normal(mu, 0.4), 1000.0)
    return lc, dv_ms / 1000.0


def collision_cloud(rA, vA, mA, rB, vB, mB, lc_min=0.10, max_frags=1500, seed=0) -> Cloud:
    rng = np.random.default_rng(seed)
    v_rel = float(np.linalg.norm(np.subtract(vA, vB)))
    m_t, m_p = max(mA, mB), min(mA, mB)
    ratio = 0.5 * m_p * (v_rel * 1000) ** 2 / (m_t * 1000)          # J per gram of target
    catastrophic = bool(ratio >= 40.0)
    m_eff = (mA + mB) if catastrophic else m_p * v_rel**2
    n_est = max(1, int(round(0.1 * m_eff**0.75 * lc_min**-1.71)))
    n = int(np.clip(n_est, 5, max_frags))
    lc, dv = _sizes_and_dv(rng, n, lc_min, "collision")
    # each fragment descends from one parent (mass-weighted) and keeps that parent's velocity
    # plus its kick: real clouds stay on/near the parents' orbits (momentum-averaged velocity
    # would put the whole cloud on a suborbital path at large crossing angles)
    from_a = rng.random(n) < mA / (mA + mB)
    v_base = np.where(from_a[:, None], np.asarray(vA, float), np.asarray(vB, float))
    r_c = (np.asarray(rA) + np.asarray(rB)) / 2
    return Cloud(np.tile(r_c, (n, 1)), v_base + _unit_vectors(rng, n) * dv[:, None], lc,
                 n_est, float(m_eff), catastrophic, v_rel, "collision")


def explosion_cloud(r, v, mass, lc_min=0.10, max_frags=1500, seed=0) -> Cloud:
    rng = np.random.default_rng(seed)
    n_est = max(1, int(round(6 * lc_min**-1.6)))
    n = int(np.clip(n_est, 5, max_frags))
    lc, dv = _sizes_and_dv(rng, n, lc_min, "explosion")
    return Cloud(np.tile(np.asarray(r), (n, 1)), np.asarray(v) + _unit_vectors(rng, n) * dv[:, None],
                 lc, n_est, float(mass), None, None, "explosion")


def iridium_cosmos_2009(gmst: float = 0.0) -> dict:
    """Approximate geometry of the 10-Feb-2009 Iridium 33 x Cosmos 2251 collision.

    Point 72.5N 97.9E, 789 km; inclinations 86.4 deg / 74.0 deg (one ascending, one
    descending) give headings ~12 deg and ~114 deg and a relative speed of ~11.6 km/s.
    """
    # gmst (rad) rotates the inertial longitude so the point sits at 72.5N 97.9E on a rotating Earth
    lat, lon, alt = np.radians(72.5), np.radians(97.9) + gmst, 789.0
    r = (R_EARTH + alt) * np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    vc = np.sqrt(MU / np.linalg.norm(r))
    vel = lambda az: vc * (np.sin(np.radians(az)) * east + np.cos(np.radians(az)) * north)
    return dict(rA=r, vA=vel(12.1), mA=560.0, rB=r, vB=vel(113.6), mB=900.0,
                name_a="IRIDIUM 33", name_b="COSMOS 2251", lat=72.5, lon=97.9, alt=alt)


# ------------------------------------------------------------------ propagation (two-body + J2)
def _accel(r: np.ndarray) -> np.ndarray:
    rn2 = (r * r).sum(axis=1, keepdims=True)
    rn = np.sqrt(rn2)
    a = -MU * r / rn**3
    z2 = r[:, 2:3] ** 2 / rn2
    k = -1.5 * J2 * MU * R_EARTH**2 / rn**5
    return a + k * r * np.hstack([1 - 5 * z2, 1 - 5 * z2, 3 - 5 * z2])


REENTRY_ALT_KM = 80.0


def propagate_cloud(r0, v0, horizon_s: float, out_step_s: float = 60.0, dt_target: float = 15.0):
    """RK4 + J2. Returns (times_s (T,), positions (T,N,3) float32 km, T includes t=0).

    A fragment whose altitude drops below REENTRY_ALT_KM is considered re-entered: it is
    frozen and reported as NaN from then on.
    """
    sub = int(np.ceil(out_step_s / dt_target))
    dt = out_step_s / sub
    r, v = np.array(r0, float), np.array(v0, float)
    dead = np.zeros(len(r), dtype=bool)
    r_dead = np.array([R_EARTH + REENTRY_ALT_KM, 0.0, 0.0])

    def snapshot():
        out = r.astype(np.float32)
        out[dead] = np.nan
        return out

    frames, times = [snapshot()], [0.0]
    for step in range(1, int(horizon_s / dt) + 1):
        a1 = _accel(r)
        r2, v2 = r + 0.5 * dt * v, v + 0.5 * dt * a1
        a2 = _accel(r2)
        r3, v3 = r + 0.5 * dt * v2, v + 0.5 * dt * a2
        a3 = _accel(r3)
        r4, v4 = r + dt * v3, v + dt * a3
        a4 = _accel(r4)
        r = r + dt / 6 * (v + 2 * v2 + 2 * v3 + v4)
        v = v + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        dead |= np.linalg.norm(r, axis=1) < R_EARTH + REENTRY_ALT_KM
        r[dead], v[dead] = r_dead, 0.0                      # freeze: keeps the maths finite
        if step % sub == 0:
            frames.append(snapshot())
            times.append(step * dt)
    return np.array(times), np.stack(frames)


def elements(r, v) -> pd.DataFrame:
    """Osculating period / perigee / apogee altitude / inclination for state vectors."""
    rn = np.linalg.norm(r, axis=1)
    energy = (v * v).sum(axis=1) / 2 - MU / rn
    a = -MU / (2 * energy)
    h = np.cross(r, v)
    e = np.linalg.norm(np.cross(v, h) / MU - r / rn[:, None], axis=1)
    return pd.DataFrame({
        "period_min": 2 * np.pi * np.sqrt(a**3 / MU) / 60,
        "perigee_km": a * (1 - e) - R_EARTH,
        "apogee_km": a * (1 + e) - R_EARTH,
        "inc_deg": np.degrees(np.arccos(h[:, 2] / np.linalg.norm(h, axis=1))),
        "ecc": e,
    })


def catalog_elements(sats) -> pd.DataFrame:
    """Same columns from TLE mean elements (used to overlay real debris clouds)."""
    n = np.array([s.no_kozai for s in sats]) / 60.0
    e = np.array([s.ecco for s in sats])
    a = (MU / n**2) ** (1 / 3)
    return pd.DataFrame({"period_min": 2 * np.pi / n / 60, "perigee_km": a * (1 - e) - R_EARTH,
                         "apogee_km": a * (1 + e) - R_EARTH,
                         "inc_deg": np.degrees([s.inclo for s in sats]), "ecc": e})


# ------------------------------------------------------------------ exposure ranking
def exposure(cat: Catalog, t0, times, R, exclude=(), radius_km=150.0, area_m2=10.0,
             v_rel_kms=10.0, chunk=120) -> pd.DataFrame:
    """Rank catalog payloads by exposure to the fragment cloud.

    At every output time, count fragments within `radius_km` of each satellite (KD-tree) ->
    local spatial density; expected impacts = sum(density * v_rel * area * dt).
    v_rel and area are ASSUMED constants, so absolute numbers are indicative; ranking is the point.
    """
    r_all, _, ok0 = orbits.propagate(cat, t0)
    fr_alt = np.linalg.norm(R[::5], axis=2) - R_EARTH
    fr_alt = fr_alt[np.isfinite(fr_alt)]
    lo, hi = np.percentile(fr_alt, 0.1), np.percentile(fr_alt, 99.9)
    perigee, apogee = cj.altitude_bands(cat)
    mask = ok0 & (cat.kinds == KIND_PAYLOAD) & (perigee <= hi + radius_km) & (apogee >= lo - radius_km)
    mask[list(exclude)] = False
    sel = np.flatnonzero(mask)
    if not len(sel):
        return pd.DataFrame()
    arr = SatrecArray([cat.sats[i] for i in sel])
    jd0, fr0 = to_jd(t0)
    dt = float(times[1] - times[0])
    vol = 4 / 3 * np.pi * radius_km**3
    peak = np.zeros(len(sel))
    in_cloud = np.zeros(len(sel))
    hits = np.zeros(len(sel))
    for c0 in range(0, len(times), chunk):
        ts = times[c0:c0 + chunk]
        e, rs, _ = arr.sgp4(np.full(len(ts), jd0), fr0 + ts / 86400.0)
        for k in range(len(ts)):
            good = e[:, k] == 0
            counts = np.zeros(len(sel))
            pts = R[c0 + k]
            pts = pts[np.isfinite(pts[:, 0])]
            if len(pts):
                counts[good] = cKDTree(pts).query_ball_point(rs[good, k, :], r=radius_km, return_length=True)
            peak = np.maximum(peak, counts)
            in_cloud += counts > 0
            hits += counts / vol * v_rel_kms * (area_m2 * 1e-6) * dt
    out = pd.DataFrame({
        "name": [cat.names[i] for i in sel], "norad": cat.norad[sel],
        "mean_alt_km": (perigee[sel] + apogee[sel]) / 2,
        "peak_fragments_within_radius": peak.astype(int),
        "minutes_in_cloud": in_cloud * dt / 60,
        "expected_impacts": hits,
    })
    return out[out.expected_impacts > 0].sort_values("expected_impacts", ascending=False).reset_index(drop=True)