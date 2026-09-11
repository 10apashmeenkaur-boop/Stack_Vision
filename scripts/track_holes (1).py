"""
Collapse per-frame detections into one row per PHYSICAL hole.

    python track_holes.py holes_video.csv

Method: BELT COORDINATES, not frame-to-frame matching.

Frame-to-frame matching fails here. The detector only matches ~92% of
holes between consecutive frames, so a track dies roughly every 12 frames
and respawns with a new ID - producing thousands of fragments for a few
hundred real holes.

Instead: run_video.py records the true per-frame displacement (phase
correlation). Subtract the CUMULATIVE displacement from every detection
and each physical hole collapses to a tight cluster of points that stays
put for as long as the hole is in view. One cluster = one hole. A frame
where the detector misses that hole simply contributes one fewer point.

Needs frames.csv (with dx/dy columns) alongside holes_video.csv.

Output:
  holes_final.csv   one row per physical hole
"""
import argparse, csv
from collections import defaultdict
import statistics as st


def load_motion(path):
    """Cumulative belt displacement at each frame."""
    cum, cx, cy = {}, 0.0, 0.0
    with open(path, newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["frame"]))
        for r in rows:
            if r.get("dx") not in (None, ""):
                cx += float(r["dx"]); cy += float(r["dy"])
            cum[int(r["frame"])] = (cx, cy)
    return cum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--motion", default="frames.csv")
    ap.add_argument("--tol", type=float, default=7.0,
                    help="cluster radius in belt-space px")
    ap.add_argument("--min-obs", type=int, default=10)
    ap.add_argument("--out", default="holes_final.csv")
    a = ap.parse_args()

    try:
        cum = load_motion(a.motion)
    except FileNotFoundError:
        raise SystemExit(f"need {a.motion} with dx/dy columns.\n"
                         "Rerun: python run_video.py clip.mp4")

    pts = []
    with open(a.csv, newline="") as f:
        for r in csv.DictReader(f):
            fi = int(r["frame"])
            dx, dy = cum.get(fi, (0.0, 0.0))
            pts.append((float(r["cx"]) - dx, float(r["cy"]) - dy,
                        float(r["width_px"]), float(r["length_px"]), fi))
    n_rows = len(pts)
    tdx, tdy = cum[max(cum)]
    print(f"input            : {n_rows} rows")
    print(f"total belt travel: dx {tdx:+.0f}  dy {tdy:+.0f} px")

    # --- peak finding in belt space (single-linkage chains; this cannot) ---
    BIN = 2.0
    hist = defaultdict(int)
    for p in pts:
        hist[(int(p[0] // BIN), int(p[1] // BIN))] += 1

    # non-maximum suppression: a peak must be the local max within --tol
    rad = max(1, int(a.tol / BIN))
    peaks = []
    for (gx, gy), c in hist.items():
        if c < 3:
            continue
        best = True
        for ax in range(-rad, rad + 1):
            for ay in range(-rad, rad + 1):
                if (ax, ay) == (0, 0):
                    continue
                o = hist.get((gx + ax, gy + ay), 0)
                if o > c or (o == c and (gx + ax, gy + ay) < (gx, gy)):
                    best = False
                    break
            if not best:
                break
        if best:
            peaks.append((gx * BIN + BIN / 2, gy * BIN + BIN / 2))
    nlab = len(peaks)
    print(f"peaks found      : {nlab}")

    # assign each detection to the nearest peak within --tol
    pgrid = defaultdict(list)
    for i, (px, py) in enumerate(peaks):
        pgrid[(int(px // a.tol), int(py // a.tol))].append(i)

    label = [-1] * n_rows
    for i, p in enumerate(pts):
        gx, gy = int(p[0] // a.tol), int(p[1] // a.tol)
        best, bd = -1, a.tol
        for ax in (-1, 0, 1):
            for ay in (-1, 0, 1):
                for k in pgrid.get((gx + ax, gy + ay), ()):
                    d = ((p[0] - peaks[k][0]) ** 2 +
                         (p[1] - peaks[k][1]) ** 2) ** .5
                    if d < bd:
                        best, bd = k, d
        label[i] = best

    groups = defaultdict(list)
    for i, L in enumerate(label):
        if L >= 0:
            groups[L].append(pts[i])

    rows = []
    for L, g in groups.items():
        if len(g) < a.min_obs:
            continue
        ws = [p[2] for p in g]
        fr = [p[4] for p in g]
        rows.append({
            "hole_id": L, "n_obs": len(g),
            "width_px": round(st.median(ws), 2),
            "width_stdev": round(st.pstdev(ws), 2) if len(ws) > 1 else 0.0,
            "length_px": round(st.median([p[3] for p in g]), 2),
            "first_frame": min(fr), "last_frame": max(fr),
            "belt_x": round(st.median([p[0] for p in g]), 1),
            "belt_y": round(st.median([p[1] for p in g]), 1),
        })

    if not rows:
        raise SystemExit("no clusters - try --tol 10 --min-obs 3")

    rows.sort(key=lambda r: (r["first_frame"], r["belt_y"], r["belt_x"]))
    for i, r in enumerate(rows, 1):
        r["hole_id"] = i

    ws = [r["width_px"] for r in rows]
    obs = [r["n_obs"] for r in rows]
    sds = [r["width_stdev"] for r in rows]
    print(f"clusters         : {nlab}  (kept {len(rows)} with "
          f">={a.min_obs} obs)")
    print(f"PHYSICAL HOLES   : {len(rows)}")
    print(f"obs per hole     : median {st.median(obs):.0f}  "
          f"range {min(obs)}-{max(obs)}")
    print(f"width px         : median {st.median(ws):.2f}  "
          f"range {min(ws):.1f}-{max(ws):.1f}")
    print(f"per-hole stdev   : median {st.median(sds):.2f} px")

    covered = sum(obs)
    print(f"\nsanity: {covered} of {n_rows} detections assigned "
          f"({100*covered/n_rows:.0f}%)")
    if covered < 0.6 * n_rows:
        print("  ^ low. raise --tol or lower --min-obs")

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {a.out}  ({len(rows)} rows instead of {n_rows})")


if __name__ == "__main__":
    main()
