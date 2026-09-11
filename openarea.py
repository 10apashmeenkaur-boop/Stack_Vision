"""
openarea.py - plate-level wear from OPEN AREA, not from counting holes.

Why this instead of per-hole detection:

  Per-hole detection finds 20-32 of 39 slots. Glare, debris and blur eat the
  rest, and recall varies plate to plate - so the average is computed over a
  different, biased subset each time. That is why P1 read 5.4 and P3 read 6.6
  on steel that wears uniformly.

  Open area needs no segmentation. A slot that is split by debris, merged
  with its neighbour, or partly glared still contributes its true dark area.
  Nothing is counted, so nothing is missed.

  mean slot width  =  open_area / total_slot_length

  Slot LENGTH does not change with wear - only width does. So with the slot
  count and length fixed by the drawing, open area maps straight to mean
  width, and it is robust to every failure mode above.

    python openarea.py clip.mp4
    python openarea.py clip.mp4 --nominal-mm 4.0 --slots 39
"""
import argparse, csv, json
import cv2
import numpy as np
import statistics as st
import gauge


def open_area(rect, close_h=41, dark_frac=0.45):
    """Fraction of the rectified plate that is open (dark)."""
    g = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
    g = cv2.medianBlur(g, 5)

    # GLOBAL threshold, not adaptive. Slots are the darkest thing on the
    # plate. Adaptive thresholding keys on LOCAL contrast, so every glare
    # streak on the bright metal comes out as "open" - which inflated open
    # area from ~14% to 24% and every width with it.
    # No CLAHE either: it boosts local contrast, i.e. it boosts the glare.
    lo = float(np.percentile(g, 2))
    hi = float(np.percentile(g, 75))       # the plate face
    thr = lo + (hi - lo) * dark_frac
    bw = (g < thr).astype(np.uint8) * 255
    # clean speckle, bridge debris inside slots
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((7, 1), np.uint8))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((close_h, 1), np.uint8))
    # drop anything not slot-shaped, so a shadow at the plate edge does not
    # count as open area
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    keep = np.zeros_like(bw)
    area = 0
    lens = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if h < 30 or h / max(w, 1) < 1.5:
            continue
        if a > 0.10 * bw.size:          # a huge blob is not a slot
            continue
        keep[lab == i] = 255
        area += a
        lens.append(h)
    # slot length from the SAME mask as the area. Taking it from the
    # per-hole detector instead gave P2 25 mm where P1/P3 gave 75-77,
    # because that detector fragments on some plates. One source, one
    # failure mode.
    med_len = float(np.median(lens)) if lens else 0.0
    return area, keep, bw, med_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--plates", default="plates.json")
    ap.add_argument("--frames", type=int, nargs="+", default=None)
    ap.add_argument("--slots", type=int, default=39,
                    help="slots per plate, from the drawing")
    ap.add_argument("--slot-len-mm", type=float, default=None,
                    help="mean slot length in mm. Default: measured from the "
                         "image, which is safe because length does not wear.")
    ap.add_argument("--nominal-mm", type=float, default=4.0,
                    help="new-plate slot width, from the drawing")
    ap.add_argument("--reject-mm", type=float, default=6.0)
    ap.add_argument("--dark", type=float, default=0.45,
                    help="0-1 between darkest and plate face. Lower = only "
                         "the very darkest counts as open.")
    a = ap.parse_args()

    cfg = json.load(open(a.plates))
    gauge.set_plate(*cfg.get("plate_mm", [315.0, 255.0]))
    plates = cfg["plates"]
    frames = a.frames or [cfg.get("frame", 100)]
    PPM = gauge.PX_PER_MM

    print(f"plates           : {len(plates)}   frames: {frames}")
    print(f"assumed          : {a.slots} slots/plate, nominal "
          f"{a.nominal_mm} mm, reject over {a.reject_mm} mm")

    cap = cv2.VideoCapture(a.video)
    rows = []
    for fno in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno - 1)
        ok, frame = cap.read()
        if not ok:
            continue
        print(f"\nframe {fno}")
        for k, pts in enumerate(plates, 1):
            Hm = gauge.homography_from_corners(pts)
            rect = gauge.rectify(frame, Hm)
            area_px, keep, _, len_px = open_area(rect, dark_frac=a.dark)

            # slot length: measured, since length does not change with wear
            res = gauge.measure_plate(rect)
            if a.slot_len_mm:
                Lmm, src = a.slot_len_mm, "given"
            elif len_px > 0:
                Lmm, src = len_px / PPM, "from mask"
            else:
                print(f"  P{k}: no slots found")
                continue

            area_mm2 = area_px / (PPM * PPM)
            total_len_mm = a.slots * Lmm
            mean_w = area_mm2 / total_len_mm

            open_frac = 100 * area_px / keep.size
            nominal_frac = 100 * (a.slots * Lmm * a.nominal_mm) / \
                           (gauge.PLATE_W_MM * gauge.PLATE_H_MM)
            wear = 100 * (mean_w - a.nominal_mm) / a.nominal_mm

            verdict = "REJECT" if mean_w > a.reject_mm else \
                      "WATCH" if mean_w > a.nominal_mm * 1.25 else "PASS"

            cv2.imwrite(f"open_P{k}_f{fno:04d}.png",
                        cv2.addWeighted(rect, 0.6,
                                        cv2.cvtColor(keep, cv2.COLOR_GRAY2BGR),
                                        0.4, 0))
            rows.append({"frame": fno, "plate": k,
                         "open_pct": round(open_frac, 2),
                         "nominal_open_pct": round(nominal_frac, 2),
                         "slot_len_mm": round(Lmm, 1),
                         "mean_width_mm": round(mean_w, 2),
                         "wear_pct": round(wear, 1),
                         "verdict": verdict,
                         "n_holes_detected": len(res["holes"])})
            print(f"  P{k}: open {open_frac:5.2f}%  (new plate would be "
                  f"{nominal_frac:5.2f}%)  slot len {Lmm:.0f} mm [{src}]")
            print(f"      mean slot width {mean_w:5.2f} mm  "
                  f"= {wear:+.0f}% vs nominal   {verdict}")
            print(f"      (per-hole method detected only "
                  f"{len(res['holes'])} of {a.slots} slots)")
    cap.release()

    if not rows:
        raise SystemExit("nothing measured")
    with open("openarea_report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    ws = [r["mean_width_mm"] for r in rows]
    print(f"\nplate mean widths : " + "  ".join(f"{x:.2f}" for x in ws))
    print(f"spread            : {max(ws)-min(ws):.2f} mm")
    print("                    ^ this is the number to judge the method by.")
    print("                      Wear is uniform, so plates should agree.")
    print("\nwrote openarea_report.csv, open_P*_f*.png")
    print("Open the PNGs: the highlighted region IS what was counted as open.")


if __name__ == "__main__":
    main()
