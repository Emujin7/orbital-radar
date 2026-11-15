"""Orbit data + propagation layer (no Streamlit code in here, so it's testable)."""
from __future__ import annotations

import io
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import requests
from sgp4.api import Satrec, SatrecArray, jday

R_EARTH = 6378.137          # km, equatorial radius
MU = 398600.4418            # km^3/s^2
CELESTRAK = "https://celestrak.org/NORAD/elements/gp.php"
CACHE_DIR = Path(tempfile.gettempdir()) / "orbital_radar_cache"      # works on any host, incl. Streamlit Cloud
DATA_DIR = Path(__file__).resolve().parent / "data"      # optional bundled snapshot (see snapshot.py)
SOURCES: dict = {}     # group -> where its data came from (shown in the app)
STATS: dict = {}       # group -> {"records": n, "skipped": n}

# label -> CelesTrak GROUP name
GROUPS = {
    "Space stations": "stations",
    "Active satellites": "active",
    "Starlink": "starlink",
    "OneWeb": "oneweb",
    "GPS": "gps-ops",
    "Debris: Fengyun-1C (2007 ASAT)": "fengyun-1c-debris",
    "Debris: Cosmos 2251 (2009 collision)": "cosmos-2251-debris",
    "Debris: Iridium 33 (2009 collision)": "iridium-33-debris",
    "Debris: Cosmos 1408 (2021 ASAT)": "cosmos-1408-debris",
}

KIND_PAYLOAD, KIND_ROCKET, KIND_DEBRIS = 0, 1, 2
KIND_LABELS = {KIND_PAYLOAD: "Payload", KIND_ROCKET: "Rocket body", KIND_DEBRIS: "Debris"}


def parse_tles(text: str) -> list[tuple[str, str, str]]:
    """Parse 3-line TLE text into (name, line1, line2) tuples."""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    out = []
    i = 0
    while i < len(lines) - 2:
        if lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            out.append((lines[i].strip(), lines[i + 1], lines[i + 2]))
            i += 3
        else:
            i += 1
    return out


def omm_to_tles(csv_text: str) -> tuple[list[tuple[str, str, str]], int]:
    """CelesTrak CSV (OMM) -> (name, line1, line2) tuples, plus how many records could not be converted.

    Since 2026-07-11 CelesTrak no longer serves objects numbered 100000+ in the TLE format, only in
    CSV/JSON/XML. We read the CSV and re-write each record as a TLE; numbers >= 100000 become
    'Alpha-5' (A0000 = 100000), which python-sgp4 and satellite.js both understand.
    Objects numbered above 339999 (analyst objects) cannot be written as TLEs and are skipped.
    """
    from sgp4 import omm
    from sgp4.exporter import export_tle

    out, skipped = [], 0
    for fields in omm.parse_csv(io.StringIO(csv_text)):
        sat = Satrec()
        try:
            omm.initialize(sat, fields)
            l1, l2 = export_tle(sat)
        except Exception:
            skipped += 1
            continue
        out.append((fields.get("OBJECT_NAME", "UNKNOWN").strip(), l1, l2))
    return out, skipped


def _snapshot_note() -> str:
    try:
        import json
        info = json.loads((DATA_DIR / "snapshot_info.json").read_text())
        return f"bundled snapshot downloaded {info['downloaded_utc']}"
    except Exception:
        return "bundled snapshot"


def fetch_tles(group: str, ttl: int = 7200, backoff: int = 1800) -> list[tuple[str, str, str]]:
    """Get a CelesTrak group (CSV/OMM format). Order of preference:

    1. a saved copy younger than `ttl` (GP data only changes every 2 hours, so we never ask more often)
    2. a live download. If the server answers anything but HTTP 200 we STOP asking for `backoff`
       seconds (CelesTrak's policy: ignoring errors gets IP addresses firewalled)
    3. an older saved copy
    4. the snapshot bundled in data/ (made with snapshot.py), so a deployed app still works when
       CelesTrak throttles shared cloud IP addresses
    """
    CACHE_DIR.mkdir(exist_ok=True)
    f, fail, snap = CACHE_DIR / f"{group}.csv", CACHE_DIR / f"{group}.fail", DATA_DIR / f"{group}.csv"
    text, source = None, ""
    if f.exists() and (time.time() - f.stat().st_mtime) < ttl:
        text, source = f.read_text(), "saved copy (less than 2 hours old)"
    else:
        blocked = fail.exists() and (time.time() - fail.stat().st_mtime) < backoff
        try:
            if blocked:
                raise RuntimeError("CelesTrak recently refused or failed this request; waiting before trying again.")
            resp = requests.get(
                CELESTRAK,
                params={"GROUP": group, "FORMAT": "csv"},
                headers={"User-Agent": "orbital-collision-radar/0.2 (student project)"},
                timeout=30,
            )
            resp.raise_for_status()
            if "NORAD_CAT_ID" not in resp.text[:600]:
                raise ValueError(f"Unexpected response: {resp.text[:80]!r}")
            text, source = resp.text, "live download from CelesTrak"
            f.write_text(text)
            if fail.exists():
                fail.unlink()
        except Exception:
            if not blocked:
                fail.write_text(str(time.time()))
            if f.exists():
                text, source = f.read_text(), "older saved copy (the download failed)"
            elif snap.exists():
                text, source = snap.read_text(), _snapshot_note()
            else:
                raise
    tles, skipped = omm_to_tles(text)
    SOURCES[group] = source
    STATS[group] = {"records": len(tles) + skipped, "skipped": skipped}
    return tles


def classify(name: str) -> int:
    n = name.upper()
    if "DEB" in n:
        return KIND_DEBRIS
    if "R/B" in n or " RB" in n:
        return KIND_ROCKET
    return KIND_PAYLOAD


@dataclass
class Catalog:
    names: list[str]
    norad: np.ndarray
    kinds: np.ndarray
    sats: list[Satrec]
    array: SatrecArray
    lines: list = field(default_factory=list)   # (line1, line2) per object, if known


def build_catalog(tles) -> Catalog:
    seen, names, ids, kinds, sats, lines = set(), [], [], [], [], []
    for name, l1, l2 in tles:
        try:
            s = Satrec.twoline2rv(l1, l2)
        except Exception:
            continue
        if s.satnum in seen:
            continue
        seen.add(s.satnum)
        names.append(name)
        ids.append(s.satnum)
        kinds.append(classify(name))
        sats.append(s)
        lines.append((l1, l2))
    return Catalog(names, np.array(ids), np.array(kinds), sats, SatrecArray(sats), lines)


def tle_lines(cat: Catalog, i: int) -> tuple[str, str]:
    """TLE lines of object i (re-exported from the elements if they weren't stored)."""
    if cat.lines and i < len(cat.lines):
        return cat.lines[i]
    from sgp4.exporter import export_tle
    return export_tle(cat.sats[i])

def to_jd(when: datetime) -> tuple[float, float]:
    return jday(when.year, when.month, when.day, when.hour, when.minute,
                when.second + when.microsecond / 1e6)


def propagate(cat: Catalog, when: datetime):
    """Positions/velocities of every object at `when` (UTC, naive or aware).

    Returns r (n,3) km and v (n,3) km/s in the TEME frame, plus an `ok` mask
    (SGP4 reports an error code for decayed / invalid elements).
    """
    jd, fr = to_jd(when)
    e, r, v = cat.array.sgp4(np.array([jd]), np.array([fr]))
    r, v = r[:, 0, :], v[:, 0, :]
    ok = (e[:, 0] == 0) & np.isfinite(r).all(axis=1)
    return r, v, ok


def orbit_path(sat: Satrec, when: datetime, n: int = 181) -> np.ndarray:
    """One full revolution of a single satellite starting at `when` (TEME, km)."""
    jd, fr = to_jd(when)
    period_min = 2 * np.pi / sat.no_kozai
    minutes = np.linspace(0, period_min, n)
    e, r, _ = sat.sgp4_array(np.full(n, jd), fr + minutes / 1440.0)
    return r[e == 0]


def orbit_summary(sat: Satrec, when: datetime) -> dict:
    n_rad_s = sat.no_kozai / 60.0
    a = (MU / n_rad_s**2) ** (1 / 3)
    jd, fr = to_jd(when)
    age = (jd + fr) - (sat.jdsatepoch + sat.jdsatepochF)
    return {
        "NORAD ID": int(sat.satnum),
        "Inclination (°)": round(float(np.degrees(sat.inclo)), 2),
        "Eccentricity": round(float(sat.ecco), 5),
        "Period (min)": round(2 * np.pi / sat.no_kozai, 2),
        "Perigee alt (km)": round(a * (1 - sat.ecco) - R_EARTH),
        "Apogee alt (km)": round(a * (1 + sat.ecco) - R_EARTH),
        "TLE age (days)": round(float(age), 2),
    }


def altitude_km(r: np.ndarray) -> np.ndarray:
    """Spherical-Earth altitude (good to ~20 km; fine for density plots)."""
    return np.linalg.norm(r, axis=1) - R_EARTH


# ------------------------------------------------------------------ frames (TEME -> lat/lon/alt)
_F = 1 / 298.257223563
_E2 = _F * (2 - _F)


def gmst_rad(jd_ut1) -> np.ndarray:
    """Greenwich mean sidereal time (IAU-82), radians. Accepts scalar or array of JD."""
    jd = np.asarray(jd_ut1, dtype=float)
    t = (jd - 2451545.0) / 36525.0
    sec = (67310.54841 + (876600 * 3600 + 8640184.812866) * t
           + 0.093104 * t**2 - 6.2e-6 * t**3)
    return (sec % 86400.0) * (2 * np.pi / 86400.0)


def teme_to_geodetic(r: np.ndarray, jd_ut1) -> np.ndarray:
    """TEME positions (n,3, km) -> (n,3) array of [lat_deg, lon_deg, alt_km] (WGS84).

    Polar motion and UT1-UTC are ignored (<~100 m effect, irrelevant for display).
    """
    r = np.atleast_2d(r)
    th = gmst_rad(jd_ut1)
    c, s = np.cos(th), np.sin(th)
    x = r[:, 0] * c + r[:, 1] * s
    y = -r[:, 0] * s + r[:, 1] * c
    z = r[:, 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - _E2))
    for _ in range(5):
        n = R_EARTH / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
        lat = np.arctan2(z + _E2 * n * np.sin(lat), p)
    n = R_EARTH / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        alt = np.where(np.abs(lat) < np.radians(80),
                       p / np.cos(lat) - n,
                       z / np.sin(lat) - n * (1 - _E2))
    return np.column_stack([np.degrees(lat), np.degrees(lon), alt])


def geodetic_now(r: np.ndarray, when: datetime) -> np.ndarray:
    jd, fr = to_jd(when)
    return teme_to_geodetic(r, jd + fr)


def orbit_path_geodetic(sat: Satrec, when: datetime, n: int = 181) -> np.ndarray:
    """One revolution as [lat, lon, alt_km] rows (ground track + altitude, Earth rotating)."""
    jd, fr = to_jd(when)
    minutes = np.linspace(0, 2 * np.pi / sat.no_kozai, n)
    frs = fr + minutes / 1440.0
    e, r, _ = sat.sgp4_array(np.full(n, jd), frs)
    ok = e == 0
    return teme_to_geodetic(r[ok], jd + frs[ok])