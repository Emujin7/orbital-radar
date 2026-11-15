"""Orbital Collision Radar.   Run:  streamlit run app.py"""
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import breakup as bk
import conjunctions as cj
import orbits
import viz

st.set_page_config(page_title="Orbital Collision Radar", page_icon="🛰️", layout="wide")

FAMOUS = ("ISS (", "CSS (", "HST", "TIANGONG")      # always drawn, so they are easy to find
RISK = {"HIGH": ("🔴", "High"), "WATCH": ("🟡", "Watch"), "LOW": ("🟢", "Low")}


@st.cache_data(ttl=7200, show_spinner="Downloading the latest orbit data…")
def load_group(group: str):
    return orbits.fetch_tles(group)


@st.cache_resource(show_spinner="Preparing satellites…")
def get_catalog(tles: tuple):
    return orbits.build_catalog(tles)


def show_html(html: str, height: int) -> None:
    """Embed an HTML page; st.iframe on new Streamlit, components.html on old."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        components.html(html, height=height)


def km_text(x: float) -> str:
    return f"{x:.2f}" if x < 1 else f"{x:.1f}"


def in_words(delta: timedelta) -> str:
    s = int(delta.total_seconds())
    if s < 0:
        return "already passed"
    h, m = divmod(s // 60, 60)
    return f"in {h} h {m} min" if h else f"in {m} min"


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🛰️ Orbital Radar")
st.sidebar.caption("A live map of satellites and space junk, and what could go wrong up there.")
chosen = st.sidebar.multiselect("What to show", list(orbits.GROUPS), default=["Active satellites"])
with st.sidebar.expander("Advanced"):
    n_show = st.slider("Max objects drawn", 500, 6000, 3000, step=250,
                       help="Fewer = smoother on slow computers.")
    exag = st.slider("Stretch heights (easier to see)", 1.0, 5.0, 2.0, step=0.5,
                     help="Orbits are very close to Earth compared to its size. Stretching makes them visible.")
    upload = st.file_uploader("Use your own orbit file (TLE)", type=["txt", "tle"])

# ---------------------------------------------------------------- data
tles = []
if upload is not None:
    tles += orbits.parse_tles(upload.getvalue().decode("utf-8", "ignore"))
for label in chosen:
    try:
        tles += load_group(orbits.GROUPS[label])
    except Exception as exc:
        st.sidebar.error(f"Couldn't download “{label}” ({exc.__class__.__name__}). "
                         "Check your internet, or upload an orbit file under Advanced.")
if not tles:
    st.title("🛰️ Orbital Collision Radar")
    st.info("Pick something to show in the sidebar (for example **Active satellites**).")
    st.stop()

cat = get_catalog(tuple(tles))
now = datetime.now(timezone.utc)
r, v, ok = orbits.propagate(cat, now)
alt = np.full(len(r), np.nan)
alt[ok] = orbits.altitude_km(r[ok])
ok_idx = np.flatnonzero(ok)
names_ok = [cat.names[i] for i in ok_idx]


def pick_pool(limit: int, must=(), seed: int = 0) -> np.ndarray:
    """Which objects to draw: the ones that matter first, then a random sample up to `limit`."""
    keep = set(int(i) for i in must) & set(ok_idx.tolist())
    keep |= {int(i) for i in ok_idx if any(k in cat.names[i] for k in FAMOUS)}
    rest = np.array([i for i in ok_idx if int(i) not in keep], dtype=int)
    room = max(limit - len(keep), 0)
    if len(rest) > room:
        rest = np.sort(np.random.default_rng(seed).choice(rest, room, replace=False))
    return np.concatenate([np.array(sorted(keep), dtype=int), rest]).astype(int)


def sky(pool, start, end, height=700, **kw):
    return viz.sky_html([orbits.tle_lines(cat, i) for i in pool], cat.kinds[pool],
                        [cat.names[i] for i in pool], start, end, exaggeration=exag, height=height, **kw)


# ---------------------------------------------------------------- header
st.title("🛰️ Orbital Collision Radar")
st.write("A live map of satellites and space junk around Earth. **Explore** them, find **close calls** "
         "between objects, and see **what would happen** if two of them collided.")
st.info("**Why this matters:** thousands of satellites and pieces of junk share Earth's orbit, moving about 7 km every second. "
        "In 2009 two satellites collided and created thousands of new pieces, which made the next collision more likely. "
        "**This app answers two questions:** which objects will pass dangerously close soon, and what would a collision "
        "do to everything else up there?")
n_ok = int(ok.sum())
n_deb = int((cat.kinds[ok] == orbits.KIND_DEBRIS).sum())
hist, edges = np.histogram(alt[ok & (alt < 2000)], bins=np.arange(0, 2020, 20))
busy = int(np.argmax(hist)) if hist.sum() else 0
m1, m2, m3 = st.columns(3)
m1.metric("Objects tracked", f"{n_ok:,}")
m2.metric("Of them, space junk", f"{n_deb:,}")
m3.metric("Most crowded height", f"{int(edges[busy])}–{int(edges[busy + 1])} km")
with st.expander("How does it work? (the method in 6 steps)"):
    st.markdown(
        "1. **Get real data:** public orbit records from CelesTrak (US Space Force catalog), cleaned and de-duplicated.\n"
        "2. **Compute positions:** the standard SGP4 orbit model turns each record into a position and speed at any moment.\n"
        "3. **Find close calls:** checking every pair would take about a billion comparisons, so we first drop objects at "
        "different heights, then use a **KD-tree** (a fast spatial search) at every 10-second step, then refine each candidate exactly.\n"
        "4. **Score the risk:** from the gap, the age of the data and its uncertainty.\n"
        "5. **Simulate a breakup:** a simplified **NASA breakup model** creates the fragments, and we track them with "
        "physics that includes Earth's bulge (J2).\n"
        "6. **Rank who is in danger:** count how many fragments pass near each satellite, and for how long.\n\n"
        "**Checked against reality:** the orbit physics matches the standard SGP4 software to within 0.3 km over 6 hours, "
        "and the 2009 replay is compared with the real debris still tracked today.")
tab_explore, tab_calls, tab_what = st.tabs(["🌍 Explore", "⚠️ Close calls", "💥 What if?"])

# ================================================================ EXPLORE
with tab_explore:
    pick = st.selectbox("🔍 Find a satellite", names_ok, index=None, key="find_sat",
                        placeholder="Type a name, for example ISS or STARLINK-1007")
    must = [cat.names.index(pick)] if pick else []
    pool = pick_pool(n_show, must)
    focus = int(np.flatnonzero(pool == must[0])[0]) if must else None
    show_html(sky(pool, now, now + timedelta(hours=12), speed=60, focus=focus, loop=True, height=700), 700)
    st.caption(f"Showing {len(pool):,} of {n_ok:,} objects. "
               "**Hover** a dot for its name · **click** it for details and its path · press **Play** to watch them move.")
    with st.expander("What am I looking at?"):
        st.markdown(
            "- Every dot is a **real object orbiting Earth**, placed using public orbit data from CelesTrak.\n"
            "- 🔵 working satellites · 🟠 used-up rocket stages · 🔴 space junk (broken or abandoned pieces)\n"
            "- Satellites move fast, about **7–8 km every second**. Press **Play** and pick a speed to see them move.\n"
            "- The orbit data is good to a few kilometres, which is enough to explore and screen, "
            "but this is **not** an official warning system.")
    with st.expander("Which heights are the most crowded?"):
        st.plotly_chart(viz.altitude_hist(alt[ok], cat.kinds[ok]), width="stretch")
        st.caption("Tall spikes are groups of satellites sharing a height, such as Starlink, "
                   "or clouds of debris left by past accidents.")
        with st.expander("Where does this data come from?"):
            age_h = np.array([(sum(orbits.to_jd(now)) - (s.jdsatepoch + s.jdsatepochF)) * 24 for s in cat.sats])
            n_new = int((cat.norad >= 100000).sum())
            st.markdown(
            "- **Source:** [CelesTrak](https://celestrak.org), run by Dr T.S. Kelso. It republishes the public catalog of the "
            "US Space Force's tracking network. Free, no account or key needed.\n"
            "- **What a record is:** a few numbers describing one orbit (shape, tilt, position at a moment). "
            "We turn them into positions with a standard model called **SGP4**.\n"
            "- **How we download it:** `celestrak.org/NORAD/elements/gp.php?GROUP=…&FORMAT=csv`. "
            "The data changes only every **2 hours**, so we keep a copy for 2 hours, and if the server ever refuses us we stop asking, "
            "as CelesTrak's rules require.\n"
            f"- **How fresh is it right now:** typical record is **{np.median(age_h):.0f} hours** old "
            f"(oldest **{age_h.max() / 24:.1f} days**).\n"
            "- **How accurate:** positions are good to roughly **1–3 km** and get worse as a record ages.\n"
            f"- **New objects:** since 11 July 2026 the old 5-digit catalog numbers ran out, and CelesTrak no longer offers newer objects in the old "
            f"TLE format. We use its CSV format instead, so newer satellites are included ({n_new:,} in what you loaded).")
    group_ids = [orbits.GROUPS[g] for g in chosen]
    downloaded = sum(orbits.STATS.get(g, {}).get("records", 0) for g in group_ids)
    skipped = sum(orbits.STATS.get(g, {}).get("skipped", 0) for g in group_ids)
    where = sorted({orbits.SOURCES[g] for g in group_ids if g in orbits.SOURCES})
    st.markdown("**Data quality report**")
    st.table(pd.DataFrame({"Check": [
            "Records downloaded", "Duplicates removed (same object in several groups)",
            "Objects the orbit model can't compute right now (decayed or invalid)",
            "Objects skipped (9-digit analyst numbers don't fit the TLE format)",
            "Typical age of a record", "Oldest record", "Where this data came from"],
            "Result": [f"{downloaded:,}" if downloaded else "n/a", f"{max(len(tles) - len(cat.names), 0):,}",
                       f"{int((~ok).sum()):,}", f"{skipped:,}", f"{np.median(age_h):.0f} hours",
                       f"{age_h.max() / 24:.1f} days", "; ".join(where) if where else "your uploaded file"]}))

# ================================================================ CLOSE CALLS
with tab_calls:
    st.subheader("Close calls")
    st.write("Which objects will pass dangerously close to each other soon? "
             "Choose a time window, then press the button.")
    a, b = st.columns(2)
    hrs = a.select_slider("Look ahead", options=[3, 6, 12, 24], value=6, format_func=lambda h: f"{h} hours")
    thr = b.select_slider("Count it as a close call if closer than", options=[2, 5, 10, 20, 50], value=10,
                          format_func=lambda k: f"{k} km")
    if n_ok > 4000:
        st.caption("⏳ With this many objects it can take a few minutes. For a quick try, choose fewer groups in the sidebar.")
    if st.button("Find close calls", type="primary"):
        bar = st.progress(0.0, text="Searching…")
        t0 = time.time()
        res = cj.screen(cat, datetime.now(timezone.utc), hours=hrs, step_s=10, threshold_km=thr,
                        min_rel_speed_kms=0.5,          # skip objects flying together (docked, same launch)
                        progress=lambda p: bar.progress(min(p, 1.0), text=f"Searching… {p:.0%}"))
        bar.empty()
        st.session_state["conj"] = (res, len(cat.names), hrs, thr, time.time() - t0)

    cs = st.session_state.get("conj")
    if cs is None:
        st.info("Nothing searched yet.")
    else:
        res, n_obj, h, th, secs = cs
        if res.empty:
            st.success(f"Good news: no two objects get closer than {th} km in the next {h} hours.")
        else:
            top = res.iloc[0]
            st.markdown(f"**{len(res)} close calls** in the next {h} hours. The closest: "
                        f"**{top.name_a}** and **{top.name_b}**, about **{km_text(top.miss_km)} km** apart.")
            for x in res.head(3).itertuples():
                emoji, word = RISK[x.risk]
                with st.container(border=True):
                    st.markdown(f"**{x.name_a}** and **{x.name_b}**")
                    st.write(f"Closest moment: **{x.tca_utc:%b %d, %H:%M} UTC** "
                             f"({in_words(x.tca_utc.to_pydatetime() - now)}). Predicted gap: about **{km_text(x.miss_km)} km**, "
                             f"but the orbit data is only good to roughly ±{x.sigma_km:.1f} km, so the real gap could differ. "
                             f"They pass each other at **{x.rel_speed_kms:.1f} km/s**. Risk: {emoji} **{word}**.")
                    if x.rel_speed_kms < 2:
                        st.caption("ℹ️ Low passing speed: these two are on very similar orbits, so this may be a gentle "
                                   "pass between neighbours rather than a head-on crossing.")
            with st.expander("How is the risk decided?"):
                st.markdown(
                    "Orbit data is only accurate to a few kilometres, and gets less accurate as it ages. "
                    "From the distance and the age of the data we estimate how likely a hit is. "
                    "🟢 **Low**: very unlikely · 🟡 **Watch**: worth keeping an eye on · 🔴 **High**: operators would consider moving. "
                    "This is a rough screening, not an official warning.\n\n"
                    "**Does “0 km” mean they will collide?** No. It means our best estimate of the two paths crosses. "
                    "Orbit data can be off by a few kilometres, so a predicted 0 km could really be 2 or 3 km, "
                    "and real operators get far more precise data before they decide anything. "
                    "To judge how serious a close call is, look at the **speed** (a high speed means a head-on crossing, "
                    "a low speed means neighbours) and the **risk label**.")

            st.caption("👇 Click a row to watch that close call happen on the globe.")
            table = res[["name_a", "name_b", "tca_utc", "miss_km", "rel_speed_kms", "risk"]]
            picked = st.dataframe(
                table, width="stretch", hide_index=True, on_select="rerun", selection_mode="single-row",
                column_config={
                    "name_a": "Object A", "name_b": "Object B",
                    "tca_utc": st.column_config.DatetimeColumn("Closest moment (UTC)", format="MMM DD, HH:mm"),
                    "miss_km": st.column_config.NumberColumn("Predicted gap (km)", format="%.2f"),
                    "rel_speed_kms": st.column_config.NumberColumn("Speed (km/s)", format="%.1f"),
                    "risk": "Risk"})
            st.download_button("Download as CSV", res.to_csv(index=False), "close_calls.csv", "text/csv")

            rows = picked.selection.rows
            if rows:
                x = res.iloc[rows[0]]
                fa, fb = np.flatnonzero(cat.norad == x.norad_a), np.flatnonzero(cat.norad == x.norad_b)
                if len(fa) and len(fb):
                    ia, ib = int(fa[0]), int(fb[0])
                    tca = x.tca_utc.to_pydatetime()
                    cp = pick_pool(1500, must=[ia, ib])
                    pos = {int(i): k for k, i in enumerate(cp)}
                    r2, _, _ = orbits.propagate(cat, tca)
                    spot = orbits.geodetic_now(r2[ia][None, :], tca)[0]
                    show_html(sky(cp, tca - timedelta(minutes=10), tca + timedelta(minutes=10), speed=60,
                                  highlight=[pos[ia], pos[ib]], view=(spot[0], spot[1]), ref_time=tca,
                                  autoplay=True, height=620), 620)
                    st.caption("The two yellow dots are the pair. The clock shows time to the closest moment (T+00:00).")
                else:
                    st.warning("These objects aren't in the groups currently loaded.")

# ================================================================ WHAT IF
with tab_what:
    st.subheader("What if?")
    st.write("See what a collision or explosion would do: how many pieces it creates, how they spread, "
             "and which satellites would be in danger.")
    scen = st.radio("Pick a scenario", ["Two objects collide", "One object explodes",
                                        "Replay the real 2009 crash"], horizontal=True)
    is_coll, is_expl = scen.startswith("Two"), scen.startswith("One")
    pa = pb = None
    row = None
    cs = st.session_state.get("conj")
    if is_coll:
        if cs is None or cs[0].empty:
            st.info("First go to **Close calls** and search. Then come back and pick one of those pairs to collide.")
        else:
            cres = cs[0]
            labels = [f"{x.name_a} and {x.name_b} · {x.miss_km:.1f} km apart · {x.tca_utc:%b %d %H:%M} UTC"
                      for x in cres.itertuples()]
            k = st.selectbox("Which close call should become a collision?", range(len(labels)),
                             format_func=lambda i: labels[i])
            row = cres.iloc[k]
            fa, fb = np.flatnonzero(cat.norad == row.norad_a), np.flatnonzero(cat.norad == row.norad_b)
            if len(fa) and len(fb):
                pa, pb = int(fa[0]), int(fb[0])
            else:
                st.warning("These objects aren't in the groups currently loaded.")
    elif is_expl:
        pick_x = st.selectbox("Which object explodes?", names_ok, index=None, key="expl_pick",
                              placeholder="Type a name, for example ISS")
        pa = cat.names.index(pick_x) if pick_x else None
    else:
        st.caption("In February 2009 two satellites, Iridium 33 and Cosmos 2251, hit each other 789 km above Siberia. "
                   "We re-create it and compare our simulated debris with the real debris still being tracked today.")

    with st.expander("Advanced settings"):
        s1, s2, s3, s4 = st.columns(4)
        lc_cm = s1.slider("Smallest piece (cm)", 5, 30, 10)
        n_cap = s2.slider("Pieces to simulate", 300, 3000, 1200, step=100)
        hrs_b = s3.slider("Hours to simulate", 3, 48, 24)
        rad = s4.slider("“Nearby” means within (km)", 50, 300, 150, step=25)
        mass_a = mass_b = None
        if pa is not None:
            mm1, mm2 = st.columns(2)
            mass_a = mm1.number_input(f"Mass of {cat.names[pa]} (kg)", 1.0, 20000.0,
                                      float(bk.DEFAULT_MASS_KG[int(cat.kinds[pa])]), key=f"ma_{pa}")
            if pb is not None:
                mass_b = mm2.number_input(f"Mass of {cat.names[pb]} (kg)", 1.0, 20000.0,
                                          float(bk.DEFAULT_MASS_KG[int(cat.kinds[pb])]), key=f"mb_{pb}")
            st.caption("Real masses aren't in orbit data, so these are guesses by object type. Change them if you know better.")

    can_run = (not (is_coll or is_expl)) or pa is not None
    if st.button("💥 Simulate", type="primary", disabled=not can_run):
        with st.spinner("Breaking things up and tracking the pieces…"):
            lc = lc_cm / 100.0
            real_el = None
            if is_coll or is_expl:
                t_ev = row.tca_utc.to_pydatetime() if is_coll else now + timedelta(minutes=5)
                r2, v2, _ = orbits.propagate(cat, t_ev)
                if is_coll:
                    cloud = bk.collision_cloud(r2[pa], v2[pa], mass_a, r2[pb], v2[pb], mass_b, lc, n_cap)
                    ev_pt = (r2[pa] + r2[pb]) / 2
                    label, excl = f"{cat.names[pa]} + {cat.names[pb]} collision", [pa, pb]
                else:
                    cloud = bk.explosion_cloud(r2[pa], v2[pa], mass_a, lc, n_cap)
                    ev_pt, label, excl = r2[pa], f"{cat.names[pa]} explosion", [pa]
                ev_geo = orbits.geodetic_now(ev_pt[None, :], t_ev)[0]
            else:
                t_ev = now + timedelta(minutes=15)
                ic = bk.iridium_cosmos_2009(gmst=float(orbits.gmst_rad(sum(orbits.to_jd(t_ev)))))
                cloud = bk.collision_cloud(ic["rA"], ic["vA"], ic["mA"], ic["rB"], ic["vB"], ic["mB"], lc, n_cap)
                ev_geo = np.array([ic["lat"], ic["lon"], ic["alt"]])
                label, excl = "2009 Iridium 33 + Cosmos 2251 crash (replay)", []
                try:
                    real = load_group("iridium-33-debris") + load_group("cosmos-2251-debris")
                    real_el = bk.catalog_elements(orbits.build_catalog(real).sats)
                except Exception:
                    real_el = None
            times, R = bk.propagate_cloud(cloud.r0, cloud.v0, hrs_b * 3600, 60.0)
            ex_df = bk.exposure(cat, t_ev, times, R, exclude=excl, radius_km=rad)
            if len(ex_df):                       # scale up if only part of the cloud was simulated
                ex_df["expected_impacts"] *= cloud.n_estimated / len(cloud.lc)
                ex_df["danger"] = (100 * ex_df.expected_impacts / ex_df.expected_impacts.max()).round().astype(int)
            st.session_state["breakup"] = dict(
                cloud=cloud, times=times, R=R, el=bk.elements(cloud.r0, cloud.v0), ex=ex_df, real_el=real_el,
                t_ev=t_ev, ev_geo=ev_geo, label=label, excl=excl, hrs=hrs_b, rad=rad, lc_cm=lc_cm)

    bs = st.session_state.get("breakup")
    if bs is None:
        st.info("Choose a scenario and press **Simulate**.")
    else:
        cloud, times, R, el, ex_df = bs["cloud"], bs["times"], bs["R"], bs["el"], bs["ex"]
        fin = np.isfinite(R[-1, :, 0])
        alive0, aliveN = int(np.isfinite(R[0, :, 0]).sum()), int(fin.sum())
        lo, hi = np.percentile(np.linalg.norm(R[-1][fin], axis=1) - orbits.R_EARTH, [5, 95]) if fin.any() else (0, 0)
        burn = float((el.perigee_km < 80).mean())
        if ex_df.empty:
            danger_txt = "No satellite in the loaded groups comes near the cloud."
        else:
            danger_txt = (f"**{len(ex_df):,} satellites** pass through the cloud. Most exposed: "
                          + ", ".join(f"**{n}**" for n in ex_df.name.head(3)) + ".")
        st.info(f"💥 **{bs['label']}** would create roughly **{cloud.n_estimated:,} pieces** bigger than "
                f"{bs['lc_cm']} cm ({len(cloud.lc):,} simulated). {burn:.0%} burn up within the first lap. "
                f"After {bs['hrs']} hours, **{aliveN / max(alive0, 1):.0%}** are still in orbit, spread between "
                f"**{lo:,.0f} and {hi:,.0f} km** up (the event happens at about {bs['ev_geo'][2]:,.0f} km). "
                + danger_txt)

        # animation
        anim_h = min(bs["hrs"], 12)
        stride = 1 if anim_h <= 6 else 2
        frames = R[::stride][: int(anim_h * 3600 / (60 * stride)) + 1]
        fsel = np.sort(np.random.default_rng(1).choice(R.shape[1], min(R.shape[1], 350), replace=False))
        parents = list(bs["excl"])
        wp = pick_pool(1200, must=parents, seed=3)
        wpos = {int(i): k for k, i in enumerate(wp)}
        show_html(sky(wp, bs["t_ev"] - timedelta(minutes=10), bs["t_ev"] + timedelta(hours=anim_h), speed=60,
                      event=dict(t0=bs["t_ev"], frames=frames[:, fsel, :], step_s=60 * stride,
                                 lat=bs["ev_geo"][0], lon=bs["ev_geo"][1], alt=bs["ev_geo"][2], label=bs["label"],
                                 hide=[wpos[i] for i in parents if i in wpos]), height=700), 700)
        st.caption(f"Press **Play**. The event happens at T+00:00 ({bs['t_ev']:%b %d, %H:%M} UTC). "
                   f"It plays the first {anim_h} hours after the event ({len(fsel)} of {len(cloud.lc):,} simulated pieces shown), "
                   "then stops. Press **Replay** to watch again. Use the slider to jump around.")

        st.markdown("#### Satellites in the most danger")
        if ex_df.empty:
            st.success("None of the loaded satellites is near the cloud.")
        else:
            st.dataframe(ex_df[["name", "mean_alt_km", "minutes_in_cloud", "peak_fragments_within_radius", "danger"]].head(10),
                         width="stretch", hide_index=True, column_config={
                "name": "Satellite", "mean_alt_km": st.column_config.NumberColumn("Height (km)", format="%.0f"),
                "minutes_in_cloud": st.column_config.NumberColumn("Minutes inside the cloud", format="%.0f"),
                "peak_fragments_within_radius": f"Most pieces nearby at once (within {bs['rad']} km)",
                "danger": st.column_config.ProgressColumn("Danger (most exposed = 100)", min_value=0, max_value=100, format="%d")})
            st.download_button("Download as CSV", ex_df.to_csv(index=False), "satellites_in_danger.csv", "text/csv")
            st.caption("Danger compares satellites with each other, based on how many pieces are near them and for how long. "
                       "It uses simple assumptions (10 m² target, 10 km/s impacts), so read it as a ranking, not a prediction.")

        if bs["real_el"] is not None:
            st.success(f"Check against reality: our model predicts about {cloud.n_estimated:,} pieces ≥ {bs['lc_cm']} cm. "
                       f"{len(bs['real_el']):,} pieces from this crash are still tracked today, 17 years later "
                       "(many have fallen out of the sky since). Open the charts below to compare the shapes.")
        with st.expander("Technical charts"):
            g1, g2 = st.columns(2)
            g1.plotly_chart(viz.gabbard(el[el.perigee_km >= 80], bs["real_el"]), width="stretch")
            g2.plotly_chart(viz.spread_fig(times, R), width="stretch")
            st.caption("Left: every breakup draws an X, with pieces ending up higher (▲) or lower (▼) than the parent, "
                       "crossing at the parent's orbit. Grey = real debris today. "
                       "Right: how the cloud's height range widens, and how many pieces are still up there.")
        with st.expander("How much can I trust this?"):
            st.markdown(
                "- Fragment counts and speeds follow a **simplified NASA breakup model**. Real events vary.\n"
                "- In our tests the orbit maths matched the standard SGP4 software to within 0.3 km over 6 hours, "
                "and reproduced the textbook drift of an orbit caused by Earth's bulge.\n"
                "- The 2009 replay got the crash speed right (11.5 vs 11.65 km/s real) and the number of pieces "
                "within a factor of about 2.\n"
                "- Not modelled: air drag, sunlight pressure, pieces hitting each other.\n"
                "- Object masses are guesses. This is a **learning and screening tool**, not an official warning system.")