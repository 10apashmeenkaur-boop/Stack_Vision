"""
video_gauge.py - run the gauge over a whole video, one result per PLATE.

The camera is fixed and the plates scroll past it. So you annotate ZONES
once - fixed quads in image space - and every plate that passes through a
zone gets measured there.

    python annotate_plates.py clip.mp4 --out zones.json     (once per camera)
    python video_gauge.py clip.mp4 --zones zones.json

How a plate becomes one row:
  - every frame, each zone is measured (open area -> mean slot width)
  - belt travel is tracked by profile correlation
  - when travel reaches one plate height, the plate in that zone has been
    replaced, so the buffered readings are flushed as ONE plate result
  - that result is the MEDIAN of every frame the plate was visible for,
    which is far steadier than any single frame

Outputs:
  plates_timeline.csv   one row per physical plate
  frames_log.csv        per-frame readings, for plotting
  video_out.mp4         annotated
"""
import argparse, csv, json
import cv2
import numpy as np
import statistics as st
import gauge
import openarea


def profile_shift(prev, cur, rng=25):
    best, bc = 0, -9.0
    for s in range(-rng, rng + 1):
        if s >= 0:
            x, y = prev[s:], cur[:len(cur) - s]
        else:
            x, y = prev[:len(prev) + s], cur[-s:]
        if len(x) < 40:
            continue
        c = float(np.dot(x, y) / len(x))
        if c > bc:
            best, bc = s, c
    return best


def vprofile(gray):
    p = gray.mean(axis=1).astype(float)
    return (p - p.mean()) / (p.std() + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--zones", default="plates.json")
    ap.add_argument("--dark", type=float, default=0.35)
    ap.add_argument("--slots", type=int, default=39)
    ap.add_argument("--nominal-mm", type=float, default=4.0)
    ap.add_argument("--reject-mm", type=float, default=6.0)
    ap.add_argument("--every", type=int, default=2, help="process every Nth frame")
    ap.add_argument("--min-obs", type=int, default=8,
                    help="a plate needs this many readings to count")
    ap.add_argument("--len-range", type=float, nargs=2, default=[55.0, 90.0],
                    help="plausible slot length mm - readings outside are "
                         "dropped as bad geometry, not bad steel")
    ap.add_argument("--no-window", action="store_true")
    a = ap.parse_args()

    cfg = json.load(open(a.zones))
    gauge.set_plate(*cfg.get("plate_mm", [315.0, 255.0]))
    zones = cfg["plates"]
    PPM = gauge.PX_PER_MM

    Hs, heights = [], []
    for z in zones:
        Hs.append(gauge.homography_from_corners(z))
        ys = [p[1] for p in z]
        heights.append(max(ys) - min(ys))
    print(f"zones            : {len(zones)}  heights {[f'{h:.0f}' for h in heights]} px")

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(3)); Hh = int(cap.get(4))
    print(f"video            : {total} frames @ {fps:.1f} fps")

    out = cv2.VideoWriter("video_out.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                          fps / a.every, (W // 2, Hh // 2))

    buf = [[] for _ in zones]          # readings for the plate now in each zone
    plate_id = [0] * len(zones)
    travel = [0.0] * len(zones)
    prevp = None
    results, flog = [], []
    idx = 0
    dys = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if (idx - 1) % a.every:
            continue

        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        band = g[int(.15 * Hh):int(.60 * Hh), int(.30 * W):int(.70 * W)]
        cp = vprofile(band)
        dy = 0.0 if prevp is None else abs(profile_shift(prevp, cp))
        prevp = cp
        dys.append(dy)

        vis = frame.copy()
        for zi, (z, Hm) in enumerate(zip(zones, Hs)):
            rect = cv2.warpPerspective(frame, Hm, (gauge.CW, gauge.CH))
            area_px, keep, _, len_px = openarea.open_area(rect, dark_frac=a.dark)
            poly = np.array(z, np.int32)

            good = False
            if len_px > 0:
                Lmm = len_px / PPM
                if a.len_range[0] <= Lmm <= a.len_range[1]:
                    mm = (area_px / (PPM * PPM)) / (a.slots * Lmm)
                    buf[zi].append(mm)
                    flog.append({"frame": idx, "zone": zi + 1,
                                 "width_mm": round(mm, 3),
                                 "slot_len_mm": round(Lmm, 1),
                                 "dy": round(dy, 2)})
                    good = True

            travel[zi] += dy
            if travel[zi] >= heights[zi] and len(buf[zi]) >= a.min_obs:
                vals = buf[zi]
                med = st.median(vals)
                plate_id[zi] += 1
                results.append({
                    "zone": zi + 1, "plate": plate_id[zi],
                    "frame_end": idx, "n_obs": len(vals),
                    "width_mm": round(med, 2),
                    "stdev_mm": round(st.pstdev(vals), 3) if len(vals) > 1 else 0,
                    "wear_pct": round(100 * (med - a.nominal_mm) / a.nominal_mm, 1),
                    "verdict": ("REJECT" if med > a.reject_mm else
                                "WATCH" if med > a.nominal_mm * 1.25 else "PASS"),
                })
                print(f"  zone {zi+1} plate {plate_id[zi]:>3}  "
                      f"{med:5.2f} mm  n={len(vals):3d}  "
                      f"sd={st.pstdev(vals) if len(vals)>1 else 0:.2f}  "
                      f"{results[-1]['verdict']}")
                buf[zi] = []
                travel[zi] = 0.0

            live = st.median(buf[zi]) if buf[zi] else None
            if live is None:
                col, tag = (120, 120, 120), f"Z{zi+1} ..."
            else:
                col = ((0, 200, 0) if live <= a.nominal_mm * 1.25 else
                       (0, 210, 255) if live <= a.reject_mm else (0, 0, 255))
                tag = f"Z{zi+1} {live:.2f}mm"
            cv2.polylines(vis, [poly], True, col, 3)
            org = (int(z[0][0]) + 6, int(z[0][1]) + 26)
            cv2.putText(vis, tag, org, 0, .7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(vis, tag, org, 0, .7, col, 1, cv2.LINE_AA)

        cv2.rectangle(vis, (0, 0), (W, 40), (25, 25, 25), -1)
        cv2.putText(vis, f"frame {idx}/{total}   belt {dy:.1f} px/f   "
                    f"plates done: {len(results)}",
                    (10, 27), 0, .8, (255, 255, 255), 2, cv2.LINE_AA)
        small = cv2.resize(vis, (W // 2, Hh // 2))
        out.write(small)
        if not a.no_window:
            cv2.imshow("hira - q to stop", small)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break

    cap.release(); out.release(); cv2.destroyAllWindows()

    if not results:
        raise SystemExit(
            "no plates completed. The belt may not have moved a full plate "
            "height in this clip.\nTry --min-obs 3, or check that slot "
            "lengths are inside --len-range.")

    with open("plates_timeline.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    with open("frames_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flog[0].keys()))
        w.writeheader(); w.writerows(flog)

    ws = [r["width_mm"] for r in results]
    rej = sum(r["verdict"] == "REJECT" for r in results)
    print(f"\nbelt speed       : {st.median(dys):.2f} px/frame")
    print(f"plates measured  : {len(results)}")
    print(f"width mm         : median {st.median(ws):.2f}  "
          f"range {min(ws):.2f}-{max(ws):.2f}")
    print(f"per-plate stdev  : median "
          f"{st.median([r['stdev_mm'] for r in results]):.3f} mm")
    print("                   ^ how steady each plate read across its frames")
    print(f"rejected         : {rej} of {len(results)}")
    print("\nwrote plates_timeline.csv, frames_log.csv, video_out.mp4")


if __name__ == "__main__":
    main()
