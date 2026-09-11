"""
measure_all.py - measure every plate you annotated, holes and all.

    python measure_all.py clip.mp4
    python measure_all.py clip.mp4 --frames 1 200 400 600
    python measure_all.py clip.mp4 --offset-mm 2.3     (after caliper check)

Reads plates.json from annotate_plates.py.

Outputs:
  detected_f####.png   plates boxed + every hole coloured by mm
  plates_report.csv    one row per plate per frame
  holes_all.csv        one row per hole
  report.txt           the summary for Hira
"""
import argparse, csv, json
import cv2
import numpy as np
import statistics as st
import gauge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--plates", default="plates.json")
    ap.add_argument("--frames", type=int, nargs="+", default=None)
    ap.add_argument("--offset-mm", type=float, default=0.0,
                    help="subtract from every width; set from a caliper "
                         "reading on a healthy plate")
    ap.add_argument("--max-val-err", type=float, default=20.0,
                    help="drop a plate whose length validation is worse "
                         "than this %%")
    a = ap.parse_args()

    cfg = json.load(open(a.plates))
    gauge.set_plate(*cfg.get("plate_mm", [315.0, 255.0]))
    plate_pts = cfg["plates"]
    frames = a.frames or [cfg.get("frame", 100)]
    print(f"plates           : {len(plate_pts)}")
    print(f"frames           : {frames}")
    print(f"reference        : {gauge.PLATE_W_MM:.0f} x {gauge.PLATE_H_MM:.0f} mm"
          f"  ->  {gauge.PX_PER_MM} px/mm")

    cap = cv2.VideoCapture(a.video)
    rows, all_holes = [], []

    for fno in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno - 1)
        ok, frame = cap.read()
        if not ok:
            print(f"  frame {fno}: unreadable")
            continue
        vis = frame.copy()
        print(f"\nframe {fno}")

        for k, pts in enumerate(plate_pts, 1):
            Hm = gauge.homography_from_corners(pts)
            res = gauge.measure_plate(gauge.rectify(frame, Hm))
            val = gauge.validate(res)
            poly = np.array(pts, np.int32)

            if not res["holes"]:
                cv2.polylines(vis, [poly], True, (120, 120, 120), 2)
                print(f"  P{k}: no holes measured")
                continue

            ws = [h["width_mm"] - a.offset_mm for h in res["holes"]]
            med = st.median(ws)
            rej = 100 * sum(w > gauge.REJECT_MM for w in ws) / len(ws)
            trusted = val["error_pct"] <= a.max_val_err
            verdict = ("REJECT" if rej > 10 else
                       "WATCH" if med > gauge.WATCH_MM else "PASS")
            col = ({"PASS": (0, 200, 0), "WATCH": (0, 210, 255),
                    "REJECT": (0, 0, 255)}[verdict] if trusted
                   else (140, 140, 140))

            cv2.polylines(vis, [poly], True, col, 3)
            tag = f"P{k} {med:.1f}mm {verdict}" if trusted else f"P{k} UNVERIFIED"
            org = (int(pts[0][0]) + 6, int(pts[0][1]) + 26)
            cv2.putText(vis, tag, org, 0, .68, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(vis, tag, org, 0, .68, col, 1, cv2.LINE_AA)

            Hinv = np.linalg.inv(Hm)
            for h in res["holes"]:
                q = Hinv @ np.array([h["cx"], h["cy"], 1.0]); q /= q[2]
                mm = h["width_mm"] - a.offset_mm
                hc = ((0, 200, 0) if mm <= gauge.WATCH_MM else
                      (0, 210, 255) if mm <= gauge.REJECT_MM else (0, 0, 255))
                cv2.circle(vis, (int(q[0]), int(q[1])), 3, hc, -1)
                r = dict(h); r.update(plate=k, frame=fno,
                                      width_mm=round(mm, 2))
                all_holes.append(r)

            rows.append({
                "frame": fno, "plate": k, "n_holes": len(ws),
                "median_mm": round(med, 2),
                "p95_mm": round(sorted(ws)[int(.95 * len(ws)) - 1], 2),
                "max_mm": round(max(ws), 2),
                "reject_pct": round(rej, 1), "verdict": verdict,
                "val_length_mm": val["median_length_mm"],
                "val_error_pct": val["error_pct"],
                "trusted": trusted,
            })
            flag = "" if trusted else "   <-- validation poor, not trusted"
            print(f"  P{k}: {len(ws):3d} holes  median {med:5.2f} mm  "
                  f"reject {rej:5.1f}%  {verdict:<6}  "
                  f"val {val['error_pct']:4.1f}%{flag}")

        cv2.imwrite(f"detected_f{fno:04d}.png", vis)
    cap.release()

    if not rows:
        raise SystemExit("nothing measured")

    with open("plates_report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open("holes_all.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_holes[0].keys()))
        w.writeheader(); w.writerows(all_holes)

    good = [r for r in rows if r["trusted"]]
    use = good or rows
    meds = [r["median_mm"] for r in use]
    spread = max(meds) - min(meds)
    lines = [
        "HIRA GRATE INSPECTION",
        "=" * 46,
        f"source            : {a.video}",
        f"plates x frames   : {len(use)} measurements",
        f"holes measured    : {len(all_holes)}",
        "",
        "CALIBRATION",
        f"  planar homography per plate, {gauge.PLATE_W_MM:.0f} x "
        f"{gauge.PLATE_H_MM:.0f} mm reference",
        f"  scale {gauge.PX_PER_MM} px/mm (1 px = {1/gauge.PX_PER_MM:.2f} mm)",
        f"  independent check: slot length vs drawing rows 62/79/62 mm",
        f"  best validation this run: "
        f"{min(r['val_error_pct'] for r in use):.1f}% off",
        "",
        "RESULT",
        f"  plate medians   : {min(meds):.2f} - {max(meds):.2f} mm "
        f"(spread {spread:.2f} mm)",
        f"  overall median  : {st.median(meds):.2f} mm",
        f"  plates flagged  : "
        f"{sum(r['verdict']=='REJECT' for r in use)} of {len(use)}",
        "",
        "LIMITS",
        "  Width is the DARK OPENING and includes the slot chamfer, so it",
        "  reads wider than a caliper on the through-slot. Set --offset-mm",
        "  from one caliper measurement on a healthy plate to correct this.",
        "  Homography is per camera position - if a camera is moved, the",
        "  length validation will jump and the run flags itself UNVERIFIED.",
        "  Lens distortion is not corrected; that needs a chessboard set.",
    ]
    if spread > 1.5:
        lines += ["", f"  WARNING: plates on one drum differ by {spread:.1f} mm.",
                  "  Wear is uniform per the plant, so this is corner",
                  "  placement or optics, not condition. Re-annotate the",
                  "  plates whose validation error is highest."]
    open("report.txt", "w").write("\n".join(lines))
    print("\n" + "\n".join(lines[11:]))
    print("\nwrote detected_f*.png, plates_report.csv, holes_all.csv, report.txt")


if __name__ == "__main__":
    main()
