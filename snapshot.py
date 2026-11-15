"""Save a one-time snapshot of the CelesTrak data into data/ so a deployed app still works if
CelesTrak is slow or throttles shared cloud IP addresses.

Run it ONCE on your own computer before deploying, then commit the data/ folder:

    python snapshot.py

It sends one polite request per group (a few seconds apart) and stops at the first error, as
CelesTrak's usage policy requires.
"""
import json
import time
from datetime import datetime, timezone

import requests

import orbits


def main() -> None:
    orbits.DATA_DIR.mkdir(exist_ok=True)
    for label, group in orbits.GROUPS.items():
        resp = requests.get(orbits.CELESTRAK, params={"GROUP": group, "FORMAT": "csv"},
                            headers={"User-Agent": "orbital-collision-radar/0.2 (student project)"}, timeout=60)
        if resp.status_code != 200 or "NORAD_CAT_ID" not in resp.text[:600]:
            print(f"STOP: {group} returned HTTP {resp.status_code}. Wait a while and try again later.")
            return
        (orbits.DATA_DIR / f"{group}.csv").write_text(resp.text)
        tles, skipped = orbits.omm_to_tles(resp.text)
        print(f"{label:42s} {len(tles):6d} objects ({skipped} skipped)")
        time.sleep(6)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    (orbits.DATA_DIR / "snapshot_info.json").write_text(json.dumps({"downloaded_utc": stamp}))
    print("Saved to data/. Commit this folder with your project.")


if __name__ == "__main__":
    main()