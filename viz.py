"""Charts + the interactive sky (globe.gl for drawing, satellite.js for live orbits)."""
from __future__ import annotations

import base64
import json
from datetime import timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from orbits import KIND_LABELS, R_EARTH

COLORS = {0: "#4fc3f7", 1: "#ffb74d", 2: "#ef5350"}


# ------------------------------------------------------------------ the interactive sky
_SKY_HTML = """
<div id="wrap" style="position:relative;width:100%;height:__H__px;background:#000;border-radius:8px;overflow:hidden;font-family:system-ui,sans-serif">
  <div id="g"></div>

  <div class="ui" style="position:absolute;left:12px;top:12px;color:#ddd;font-size:12px;background:rgba(0,0,0,.6);
              padding:8px 10px;border-radius:6px;line-height:1.7">
    <span style="color:#4fc3f7">&#9679;</span> Working satellite &nbsp;
    <span style="color:#ffb74d">&#9679;</span> Rocket stage &nbsp;
    <span style="color:#ef5350">&#9679;</span> Space junk
    <span id="fragkey" style="display:none">&nbsp;<span style="color:#ff8a50">&#9679;</span> New fragments</span><br>
    <span id="stats" style="opacity:.85"></span><br>
    <span style="opacity:.6">Hover a dot for its name &middot; click it for details</span>
  </div>

  <div id="card" class="ui" style="display:none;position:absolute;right:12px;top:12px;width:250px;color:#eee;font-size:13px;
              background:rgba(10,14,25,.88);border:1px solid #3a4a6a;padding:12px 14px;border-radius:8px;line-height:1.55">
    <div style="display:flex;justify-content:space-between;gap:8px">
      <b id="cname" style="font-size:14px"></b>
      <span id="cclose" style="cursor:pointer;opacity:.7">&#10005;</span>
    </div>
    <div id="ckind" style="color:#fff176;margin-bottom:4px"></div>
    <div id="cbody"></div>
    <div style="opacity:.6;font-size:11px;margin-top:6px">Yellow line = where it goes during its next lap.</div>
  </div>

  <div id="tip" style="display:none;position:absolute;pointer-events:none;color:#fff;font-size:12px;
              background:rgba(0,0,0,.8);padding:4px 8px;border-radius:4px;white-space:nowrap"></div>

  <div class="ui" style="position:absolute;left:12px;right:12px;bottom:12px;display:flex;gap:10px;align-items:center;
              background:rgba(0,0,0,.65);padding:8px 12px;border-radius:8px;color:#eee;font-size:13px">
    <button id="play" style="padding:6px 14px;border:0;border-radius:6px;background:#2e7df6;color:#fff;
            font-weight:600;cursor:pointer;min-width:90px">&#9654; Play</button>
    <select id="speed" style="padding:5px;border-radius:6px;background:#222;color:#eee;border:1px solid #444"></select>
    <input id="seek" type="range" min="0" max="1000" value="0" style="flex:1">
    <span id="clock" style="font-variant-numeric:tabular-nums;min-width:235px;text-align:right"></span>
  </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/satellite.js@5.0.0/dist/satellite.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/globe.gl@2"></script>
<script>
const D = __DATA__;
const COL = ['#4fc3f7', '#ffb74d', '#ef5350'];
const SIZE = [0.7, 1.1, 0.5];
const KIND_NAME = ['Working satellite', 'Rocket stage', 'Space junk'];
const KIND_TEXT = ['An active satellite or a space-station part.',
                   'A used-up rocket stage left in orbit.',
                   'A broken or abandoned piece, tracked so it can be avoided.'];
const K = D.ex / 6371.0;
const N = D.tles.length;
const startMs = Date.parse(D.start), endMs = Date.parse(D.end);
const refMs = D.ref ? Date.parse(D.ref) : null;
let simMs = startMs, playing = false, speed = D.speed, sel = -1, interacting = false;
const hlSet = new Set(D.hl || []);

const recs = D.tles.map(t => satellite.twoline2satrec(t[0], t[1]));
const curLat = new Float64Array(N), curLng = new Float64Array(N), curAlt = new Float64Array(N);
const alive = new Uint8Array(N);
let selSpeed = NaN;

function b64ToU16(b64) {
  const bin = atob(b64), u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Uint16Array(u8.buffer);
}
let EV = null;
if (D.ev) {
  EV = Object.assign({}, D.ev);
  EV.t0ms = Date.parse(D.ev.t0);
  EV.u = b64ToU16(D.ev.b64);
  EV.hideSet = new Set(D.ev.hide);
  document.getElementById('fragkey').style.display = 'inline';
}
const relMs = EV ? EV.t0ms : refMs;

const wrap = document.getElementById('wrap');
const world = Globe()(document.getElementById('g'))
  .width(wrap.clientWidth).height(__H__)
  .globeImageUrl('https://cdn.jsdelivr.net/npm/three-globe/example/img/earth-blue-marble.jpg')
  .bumpImageUrl('https://cdn.jsdelivr.net/npm/three-globe/example/img/earth-topology.png')
  .backgroundImageUrl('https://cdn.jsdelivr.net/npm/three-globe/example/img/night-sky.png')
  .atmosphereColor('#5b9bff').atmosphereAltitude(0.2)
  .particlesList('pts').particleLat('lat').particleLng('lng').particleAltitude('alt')
  .particlesColor('color').particlesSize('size').particlesSizeAttenuation(true)
  .labelLat('lat').labelLng('lng').labelAltitude('alt').labelText('text')
  .labelSize(1.4).labelColor(() => '#fff176').labelDotRadius(0.5).labelResolution(2)
  .pathPoints('p').pathPointLat(p => p[0]).pathPointLng(p => p[1]).pathPointAlt(p => p[2] * K)
  .pathColor(() => '#fff176').pathStroke(1.5).pathTransitionDuration(0)
  .ringLat('lat').ringLng('lng').ringMaxRadius(7).ringPropagationSpeed(3).ringRepeatPeriod(1400).ringResolution(24)
  .ringColor(() => t => 'rgba(255,112,67,' + (1 - t) + ')');

// Stable data objects: the globe library re-uses a mesh (and its GPU shader) only when it is handed the
// SAME object again. Handing it new objects every frame makes it rebuild shaders over and over = stutter.
const HIDDEN = {lat: 0, lng: 0, alt: -1};                       // a point at Earth's centre: never visible
const PL = COL.map((c, k) => ({pts: [HIDDEN], color: c, size: SIZE[k]}));
const PL_HL = {pts: [HIDDEN], color: '#fff176', size: 3.2};
const PL_FR = {pts: [HIDDEN], color: '#ff8a50', size: 1.8};
const PLAYERS = PL.concat([PL_HL, PL_FR]);
const LBL = [0, 1, 2, 3].map(() => ({lat: 0, lng: 0, alt: -1, text: '.'}));
const HIDDEN_PATH = [[0, 0, -1 / K], [0, 0, -1 / K]];
const PATH = {p: HIDDEN_PATH};
world.particlesData(PLAYERS.slice());
world.labelsData(LBL.slice());
world.pathsData([PATH]);

// smooth, immediate controls: no coasting after you let go, lighter rendering
const ctl = world.controls();
ctl.autoRotate = false;
ctl.enableDamping = false;
ctl.addEventListener('start', () => { interacting = true; document.getElementById('tip').style.display = 'none'; });
ctl.addEventListener('end', () => { interacting = false; });
world.renderer().setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
window.addEventListener('resize', () => world.width(wrap.clientWidth));

const view = EV ? [EV.lat, EV.lon] : (D.view || [20, 0]);
world.pointOfView({lat: view[0], lng: view[1], altitude: (EV || D.view) ? 1.9 : 2.4});

function geo(pos, gm) {
  const g = satellite.eciToGeodetic(pos, gm);
  return {lat: satellite.degreesLat(g.latitude), lng: satellite.degreesLong(g.longitude), alt: g.height * K};
}

// ---- selection card + orbit path
const card = document.getElementById('card'), tip = document.getElementById('tip');
let pathSel = -1, pathAt = 0, pathWall = 0, labelKey = '', labelAt = 0;
function buildPath() {
  const rec = recs[sel], periodMs = 2 * Math.PI / rec.no * 60000, pts = [], n = 100;
  for (let k = 0; k <= n; k++) {
    const d = new Date(simMs + k / n * periodMs), pv = satellite.propagate(rec, d);
    if (!pv.position) continue;
    const g = satellite.eciToGeodetic(pv.position, satellite.gstime(d));
    pts.push([satellite.degreesLat(g.latitude), satellite.degreesLong(g.longitude), g.height]);
  }
  PATH.p = pts; world.pathsData([PATH]);
  pathSel = sel; pathAt = simMs; pathWall = performance.now();
}
function updateCard() {
  if (sel < 0) { card.style.display = 'none'; return; }
  card.style.display = 'block';
  document.getElementById('cname').textContent = D.names[sel];
  document.getElementById('ckind').textContent = KIND_NAME[D.kind[sel]];
  if (!alive[sel]) { document.getElementById('cbody').textContent = 'No valid position at this time.'; return; }
  const lapMin = 2 * Math.PI / recs[sel].no;
  document.getElementById('cbody').innerHTML = KIND_TEXT[D.kind[sel]] + '<br>' +
    '<b>' + Math.round(curAlt[sel]).toLocaleString() + ' km</b> above Earth<br>' +
    'Moving at <b>' + selSpeed.toFixed(1) + ' km/s</b> (' + Math.round(selSpeed * 3600).toLocaleString() + ' km/h)<br>' +
    'One lap around Earth every <b>' + Math.round(lapMin) + ' min</b>';
}
function select(i, fly) {
  sel = i; pathSel = -1;
  render();
  if (fly && alive[i]) world.pointOfView({lat: curLat[i], lng: curLng[i], altitude: 1.6}, 900);
}
document.getElementById('cclose').addEventListener('click', () => { sel = -1; render(); });

// ---- drawing one frame
let ringUntil = 0, prevMs = simMs;
function render() {
  const date = new Date(simMs), gm = satellite.gstime(date);
  const after = EV && simMs >= EV.t0ms;
  const sets = [[], [], []], hl = [], labels = [];
  for (let i = 0; i < N; i++) {
    alive[i] = 0;
    const parent = EV && EV.hideSet.has(i);
    if (parent && after) continue;                       // parents are destroyed
    const pv = satellite.propagate(recs[i], date);
    if (!pv.position) continue;
    const g = satellite.eciToGeodetic(pv.position, gm);
    const lat = satellite.degreesLat(g.latitude), lng = satellite.degreesLong(g.longitude);
    if (!isFinite(lat) || !isFinite(g.height)) continue;
    curLat[i] = lat; curLng[i] = lng; curAlt[i] = g.height; alive[i] = 1;
    const p = {lat: lat, lng: lng, alt: g.height * K};
    if (i === sel) selSpeed = Math.hypot(pv.velocity.x, pv.velocity.y, pv.velocity.z);
    if (parent || i === sel || hlSet.has(i)) { hl.push(p); labels.push({lat: lat, lng: lng, alt: p.alt, text: D.names[i]}); }
    else sets[D.kind[i]].push(p);
  }
  let nFrag = 0;
  const fr = [];
  if (after) {
    const f = (simMs - EV.t0ms) / (EV.step * 1000);
    const i = Math.max(0, Math.min(Math.floor(f), EV.F - 2)), w = Math.max(0, Math.min(f - i, 1));
    const u = EV.u, M = EV.N, dec = v => v === 65535 ? NaN : v / EV.scale - EV.lim;
    for (let n = 0; n < M; n++) {
      const a = (i * M + n) * 3, b = ((i + 1) * M + n) * 3;
      const x0 = dec(u[a]), y0 = dec(u[a + 1]), z0 = dec(u[a + 2]);
      if (isNaN(x0)) continue;                           // burned up
      const x1 = dec(u[b]), y1 = dec(u[b + 1]), z1 = dec(u[b + 2]);
      const lerp = (p, q) => isNaN(q) ? p : p + (q - p) * w;
      fr.push(geo({x: lerp(x0, x1), y: lerp(y0, y1), z: lerp(z0, z1)}, gm));
    }
    nFrag = fr.length;
  }
  for (let k = 0; k < 3; k++) PL[k].pts = sets[k].length ? sets[k] : [HIDDEN];
  PL_HL.pts = hl.length ? hl : [HIDDEN];
  PL_FR.pts = fr.length ? fr : [HIDDEN];
  world.particlesData(PLAYERS.slice());

  // labels: only refresh when the set changes, or (for labels on moving dots) at most 4x per second
  if (EV) labels.push({lat: EV.lat, lng: EV.lon, alt: EV.alt * K, text: EV.label});
  const lkey = labels.map(l => l.text).join('|');
  const movingLabels = labels.length > (EV ? 1 : 0);           // anything besides the fixed event label
  if (lkey !== labelKey || (movingLabels && performance.now() - labelAt > 250)) {
    for (let k = 0; k < LBL.length; k++) {
      const src = labels[k], L = LBL[k];
      if (src) { L.lat = src.lat; L.lng = src.lng; L.alt = src.alt; L.text = src.text; }
      else { L.lat = 0; L.lng = 0; L.alt = -1; L.text = '.'; }
    }
    world.labelsData(LBL.slice()); labelKey = lkey; labelAt = performance.now();
  }

  if (sel >= 0 && alive[sel]) { if (pathSel !== sel || (Math.abs(simMs - pathAt) > 120000 && performance.now() - pathWall > 1000)) buildPath(); }
  else if (pathSel !== -1) { PATH.p = HIDDEN_PATH; world.pathsData([PATH]); pathSel = -1; }

  if (EV && prevMs < EV.t0ms && simMs >= EV.t0ms && simMs - prevMs < 6 * 3600e3) {
    ringUntil = performance.now() + 6000;
    world.ringsData([{lat: EV.lat, lng: EV.lon}]);
  } else if (ringUntil && performance.now() > ringUntil) { ringUntil = 0; world.ringsData([]); }
  prevMs = simMs;

  document.getElementById('seek').value = Math.round(1000 * (simMs - startMs) / (endMs - startMs));
  let rel = '';
  if (relMs !== null) {
    const d = Math.round((simMs - relMs) / 1000), s = d < 0 ? '-' : '+', a = Math.abs(d);
    const p2 = n => String(n).padStart(2, '0');
    rel = '  (T' + s + p2(Math.floor(a / 3600)) + ':' + p2(Math.floor(a % 3600 / 60)) + ':' + p2(a % 60) + ')';
  }
  document.getElementById('clock').textContent = date.toISOString().slice(0, 19).replace('T', ' ') + ' UTC' + rel;
  document.getElementById('stats').textContent = N.toLocaleString() + ' objects' +
    (EV ? ' \\u00b7 ' + (after ? nFrag.toLocaleString() + ' fragments alive' : 'collision pending') : '');
  updateCard();
}

// ---- find the object under the mouse (hover name / click details)
function pickAt(px, py, maxPx) {
  const cam = world.camera().position;
  let best = -1, bd = maxPx * maxPx;
  for (let i = 0; i < N; i++) {
    if (!alive[i]) continue;
    const s = world.getScreenCoords(curLat[i], curLng[i], curAlt[i] * K);
    const dx = s.x - px, dy = s.y - py, d2 = dx * dx + dy * dy;
    if (d2 >= bd) continue;
    const r = 100 * (1 + curAlt[i] * K), phi = (90 - curLat[i]) * Math.PI / 180, th = (90 - curLng[i]) * Math.PI / 180;
    const x = r * Math.sin(phi) * Math.cos(th), y = r * Math.cos(phi), z = r * Math.sin(phi) * Math.sin(th);
    if (x * cam.x + y * cam.y + z * cam.z < 100 * 100) continue;   // on the far side of Earth
    best = i; bd = d2;
  }
  return best;
}
let down = null, lastHover = 0;
wrap.addEventListener('pointerdown', e => { down = {x: e.clientX, y: e.clientY}; });
wrap.addEventListener('pointerup', e => {
  if (down && Math.hypot(e.clientX - down.x, e.clientY - down.y) < 5 && !(e.target.closest && e.target.closest('.ui'))) {
    const r = wrap.getBoundingClientRect(), i = pickAt(e.clientX - r.left, e.clientY - r.top, 14);
    if (i >= 0) select(i, false);
  }
  down = null;
});
wrap.addEventListener('pointermove', e => {
  if (e.buttons || interacting) { tip.style.display = 'none'; return; }
  const now = performance.now();
  if (now - lastHover < 90) return;
  lastHover = now;
  const r = wrap.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, i = pickAt(x, y, 10);
  if (i < 0) { tip.style.display = 'none'; return; }
  tip.textContent = D.names[i] + '  \\u00b7  ' + KIND_NAME[D.kind[i]] + '  \\u00b7  ' + Math.round(curAlt[i]).toLocaleString() + ' km up';
  tip.style.left = (x + 14) + 'px'; tip.style.top = (y + 14) + 'px'; tip.style.display = 'block';
});

// ---- playback controls
const speedSel = document.getElementById('speed');
const SPEEDS = [[1, 'Real time'], [60, '1 min / sec'], [300, '5 min / sec'], [1200, '20 min / sec'], [3600, '1 hour / sec']];
if (!SPEEDS.some(s => s[0] === speed)) SPEEDS.push([speed, speed + 'x']);
SPEEDS.forEach(s => {
  const o = document.createElement('option'); o.value = s[0]; o.textContent = s[1]; speedSel.appendChild(o);
});
speedSel.value = String(speed);
speedSel.addEventListener('change', () => { speed = Number(speedSel.value); });
const playBtn = document.getElementById('play');
function setPlaying(p) { playing = p; playBtn.innerHTML = p ? '&#10074;&#10074; Pause' : '&#9654; Play'; }
playBtn.addEventListener('click', () => {
  if (!playing && simMs >= endMs) simMs = startMs;
  setPlaying(!playing);
});
document.getElementById('seek').addEventListener('input', e => {
  simMs = startMs + Number(e.target.value) / 1000 * (endMs - startMs); render();
});

let last = null, lastRender = 0, ema = 16, frames = 0, slow = 0, minGap = 70, pr = Math.min(window.devicePixelRatio || 1, 1.5);
function tick(ts) {
  if (last !== null) ema = 0.9 * ema + 0.1 * (ts - last);
  if (playing) {
    if (simMs >= endMs) {
      if (D.loop) { simMs = startMs; render(); lastRender = ts; }      // Explore: keep going forever
      else { simMs = endMs; setPlaying(false); playBtn.innerHTML = '&#8635; Replay'; render(); }
    }
    else {
      if (last !== null) simMs += (ts - last) * speed;
      if (ts - lastRender > (interacting ? 250 : minGap)) { render(); lastRender = ts; }
    }
    // light safety net: if frames stay slow for a while, lower the picture resolution a little
    if (++frames % 30 === 0) {
      slow = ema > 40 ? slow + 1 : 0;
      if (slow >= 3 && pr > 1) { pr = Math.max(1, pr - 0.25); world.renderer().setPixelRatio(pr); slow = 0; }
    }
  }
  last = ts;
  requestAnimationFrame(tick);
}
render();
if (D.focus !== null && D.focus !== undefined) select(D.focus, true);
if (D.autoplay) setPlaying(true);
requestAnimationFrame(tick);
</script>
"""


def _encode_frames(frames: np.ndarray):
    """(F,N,3) float km (NaN allowed) -> base64 uint16 + (lim, scale)."""
    lim = float(np.nanmax(np.abs(frames))) * 1.01
    scale = 65000.0 / (2 * lim)
    q = np.where(np.isfinite(frames), np.round((frames + lim) * scale), 65535).astype("<u2")
    return base64.b64encode(q.tobytes()).decode("ascii"), lim, scale


def sky_html(tles, kinds, names, start, end, speed: float = 60.0, event=None, highlight=(),
             focus=None, view=None, ref_time=None, autoplay: bool = False, loop: bool = False,
             exaggeration: float = 2.0, height: int = 700) -> str:
    """Interactive globe with live orbits (satellite.js), hover names, click details, playback.

    tles / kinds / names   parallel lists for the objects to draw (kinds: 0 payload, 1 rocket, 2 debris)
    start / end            timezone-aware datetimes bounding the playback
    highlight              indices (into tles) drawn in yellow with a name label
    focus                  index to select (card + orbit path) when the page loads
    view                   (lat, lon) to point the camera at; ref_time shows a T-/T+ clock
    loop                   restart from the beginning when the end is reached (otherwise stop with a Replay button)
    event                  optional breakup: dict(t0, frames (F,N,3) TEME km with NaN = burned up,
                           step_s, lat, lon, alt, label, hide=[indices of destroyed objects])
    """
    iso = lambda d: d.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    data = {"tles": [[a, b] for a, b in tles], "kind": [int(k) for k in kinds], "names": list(names),
            "start": iso(start), "end": iso(end), "speed": float(speed), "ex": float(exaggeration),
            "hl": [int(i) for i in highlight], "focus": None if focus is None else int(focus),
            "view": None if view is None else [float(view[0]), float(view[1])],
            "ref": iso(ref_time) if ref_time is not None else None, "autoplay": bool(autoplay),
            "loop": bool(loop), "ev": None}
    if event is not None:
        b64, lim, scale = _encode_frames(np.asarray(event["frames"]))
        F, N = event["frames"].shape[:2]
        data["ev"] = {"t0": iso(event["t0"]), "step": float(event["step_s"]), "F": int(F), "N": int(N),
                      "b64": b64, "lim": lim, "scale": scale, "hide": [int(i) for i in event.get("hide", [])],
                      "lat": float(event["lat"]), "lon": float(event["lon"]), "alt": float(event["alt"]),
                      "label": event.get("label", "")}
    blob = json.dumps(data).replace("</", "<\\/")
    return _SKY_HTML.replace("__H__", str(height)).replace("__DATA__", blob)


# ------------------------------------------------------------------ charts
def altitude_hist(alt, kinds) -> go.Figure:
    fig = go.Figure()
    for k, label in KIND_LABELS.items():
        m = (kinds == k) & (alt < 2200)
        if m.any():
            fig.add_trace(go.Histogram(x=alt[m], name=label, xbins=dict(size=20), marker_color=COLORS[k]))
    fig.update_layout(barmode="stack", template="plotly_dark", height=420,
                      xaxis_title="Height above Earth (km)", yaxis_title="Number of objects",
                      title="How crowded is each height?")
    return fig


def gabbard(sim: pd.DataFrame, real: "pd.DataFrame | None" = None) -> go.Figure:
    """Period vs apogee/perigee altitude: the classic debris-cloud signature (an X shape)."""
    fig = go.Figure()
    if real is not None and len(real):
        fig.add_trace(go.Scatter(x=real.period_min, y=real.apogee_km, mode="markers", name="Real debris: highest point",
                                 marker=dict(size=4, color="#9e9e9e", symbol="triangle-up", opacity=0.45)))
        fig.add_trace(go.Scatter(x=real.period_min, y=real.perigee_km, mode="markers", name="Real debris: lowest point",
                                 marker=dict(size=4, color="#bdbdbd", symbol="triangle-down", opacity=0.45)))
    fig.add_trace(go.Scatter(x=sim.period_min, y=sim.apogee_km, mode="markers", name="Simulated: highest point",
                             marker=dict(size=4, color="#ef5350", symbol="triangle-up", opacity=0.7)))
    fig.add_trace(go.Scatter(x=sim.period_min, y=sim.perigee_km, mode="markers", name="Simulated: lowest point",
                             marker=dict(size=4, color="#4fc3f7", symbol="triangle-down", opacity=0.7)))
    fig.update_layout(template="plotly_dark", height=430, title="Gabbard diagram (the 'X' every breakup draws)",
                      xaxis_title="Time for one lap (min)", yaxis_title="Height above Earth (km)")
    return fig


def spread_fig(times, R) -> go.Figure:
    """How the cloud's height range and population evolve over time."""
    alt = np.linalg.norm(R, axis=2) - R_EARTH
    alt[~np.isfinite(R[:, :, 0])] = np.nan
    hrs = np.asarray(times) / 3600.0
    with np.errstate(all="ignore"):
        p5, p50, p95 = (np.nanpercentile(alt, q, axis=1) for q in (5, 50, 95))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hrs, y=p95, name="Highest 5% of pieces", line=dict(color="#ef5350")))
    fig.add_trace(go.Scatter(x=hrs, y=p50, name="Typical piece", line=dict(color="#fff176")))
    fig.add_trace(go.Scatter(x=hrs, y=p5, name="Lowest 5% of pieces", line=dict(color="#4fc3f7")))
    fig.add_trace(go.Scatter(x=hrs, y=np.isfinite(R[:, :, 0]).sum(axis=1), name="Pieces still in orbit",
                             yaxis="y2", line=dict(color="#9e9e9e", dash="dot")))
    fig.update_layout(template="plotly_dark", height=380, title="How the cloud spreads out",
                      xaxis_title="Hours after the event", yaxis_title="Height above Earth (km)",
                      yaxis2=dict(title="Pieces", overlaying="y", side="right", showgrid=False))
    return fig