"""
autozone.py - find the zones automatically. No clicking.

The idea (credit: your friend): don't try to DETECT plates - search for the
region where the MEASUREMENT scores best. We already have an objective
function: slot length must come out near the drawing's 62-79 mm bands. A
quad that is on a real plate scores well; one that straddles two plates or
sits on a seam scores badly.

  vertical seams   -> found reliably by morphology (long dark lines)
  vertical extent  -> searched, scored against the drawing

Earlier attempts failed because I tried to detect the HORIZONTAL seams too.
This is a fisheye lens, so those are curved and no straight-line detector
finds them. Searching sidesteps that entirely.

    python autozone.py clip.mp4
    python autozone.py clip.mp4 --frame 100 --out zones.json

Then:
    python video_gauge.py clip.mp4 --zones zones.json
"""
import argparse, json
import cv2
import numpy as np
import statistics as st
import gauge
import openarea

TARGET_LEN = (62.0, 79.0)          # the drawing's slot row bands


def vertical_seams(frame, roi, min_w=90, seam_len=220):
    x0, y0, x1, y1 = roi
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
    g = cv2.GaussianBlur(g, (5, 5), 0)
    dark = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 61, 5)
    vert = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT,
                                                      (1, seam_len)))
    prof = cv2.GaussianBlur(vert.mean(axis=0).reshape(-1, 1).astype(np.float32),
                            (1, 9), 0).ravel()
    thr = max(float(prof.mean()) * 1.5, 1.0)
    peaks = []
    for i in range(1, len(prof) - 1):
        if prof[i] > thr and prof[i] >= prof[i-1] and prof[i] >= prof[i+1]:
            if peaks and i - peaks[-1] < min_w:
                if prof[i] > prof[peaks[-1]]:
                    peaks[-1] = i
            else:
                peaks.append(i)
    return [x0 + p for p in peaks]


def score(frame, quad, dark_frac, slots):
    """How plate-like is this quad? Uses the drawing as the judge."""
    Hm = gauge.homography_from_corners(quad)
    rect = cv2.warpPerspective(frame, Hm, (gauge.CW, gauge.CH))
    area, keep, _, len_px = openarea.open_area(rect, dark_frac=dark_frac)
    if len_px <= 0 or area <= 0:
        return -1, None
    L = len_px / gauge.PX_PER_MM
    # closeness to the nearer drawing band
    best = min(TARGET_LEN, key=lambda t: abs(t - L))
    len_err = abs(L - best) / best
    if len_err > 0.35:
        return -1, None
    # open fraction should be plausible for a grate
    frac = area / keep.size
    if not (0.06 <= frac <= 0.32):
        return -1, None
    n, _, stats, _ = cv2.connectedComponentsWithStats(keep, 8)
    count_score = min(1.0, (n - 1) / float(slots))
    s = (1.0 - len_err) * 0.65 + count_score * 0.35
    mm = (area / (gauge.PX_PER_MM ** 2)) / (slots * L)
    return s, {"slot_len_mm": round(L, 1), "open_pct": round(100 * frac, 2),
               "n_blobs": n - 1, "mean_width_mm": round(mm, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--frame", type=int, default=100)
    ap.add_argument("--out", default="zones.json")
    ap.add_argument("--dark", type=float, default=0.35)
    ap.add_argument("--slots", type=int, default=39)
    ap.add_argument("--plate-mm", type=float, nargs=2, default=[315.0, 255.0])
    ap.add_argument("--max-zones", type=int, default=3)
    ap.add_argument("--min-score", type=float, default=0.55)
    a = ap.parse_args()

    gauge.set_plate(*a.plate_mm)
    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame - 1)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("cannot read frame")
    Hh, W = frame.shape[:2]

    roi = (int(.12 * W), int(.04 * Hh), int(.82 * W), int(.78 * Hh))
    vx = vertical_seams(frame, roi)
    print(f"vertical seams   : {len(vx)} at x = {vx}")
    if len(vx) < 2:
        raise SystemExit("too few seams - the ROI may be wrong for this camera")

    cols = []
    for i in range(len(vx) - 1):
        xa, xb = vx[i], vx[i + 1]
        if xb - xa >= 80:
            cols.append((xa, xb))

    if not cols:
        raise SystemExit("no valid columns between seams")

    step = max(6, Hh // 90)
    best_row = (-1, None, None, None)

    for h in range(int(.10 * Hh), int(.34 * Hh), step):
        for y in range(roi[1], roi[3] - h, step):
            scores = []
            infos = []
            for xa, xb in cols:
                quad = [[xa, y], [xb, y], [xb, y + h], [xa, y + h]]
                s, info = score(frame, quad, a.dark, a.slots)
                scores.append(s)
                infos.append(info)
            valid = [s for s in scores if s > 0]
            if len(valid) == len(cols):
                avg_s = sum(valid) / len(valid)
                if avg_s > best_row[0]:
                    best_row = (avg_s, y, h, (scores, infos))

    if best_row[0] < a.min_score:
        for h in range(int(.10 * Hh), int(.34 * Hh), step):
            for y in range(roi[1], roi[3] - h, step):
                scores = []
                infos = []
                for xa, xb in cols:
                    quad = [[xa, y], [xb, y], [xb, y + h], [xa, y + h]]
                    s, info = score(frame, quad, a.dark, a.slots)
                    scores.append(s)
                    infos.append(info)
                valid = [s for s in scores if s > 0]
                if len(valid) >= 1:
                    avg_s = sum(valid) / len(valid)
                    if avg_s > best_row[0]:
                        best_row = (avg_s, y, h, (scores, infos))

    if best_row[0] < a.min_score or best_row[1] is None:
        raise SystemExit("nothing scored well enough. Lower --min-score, or "
                         "annotate by hand with annotate_plates.py")

    y, h = best_row[1], best_row[2]
    scores, infos = best_row[3]
    keep = []
    for idx, ((xa, xb), s, info) in enumerate(zip(cols, scores, infos)):
        if s >= a.min_score:
            q = [[xa, y], [xb, y], [xb, y + h], [xa, y + h]]
            keep.append((s, q, info))
            print(f"  column {idx+1} (x {xa}-{xb}): score {s:.2f}  {info}")

    if not keep:
        raise SystemExit("no columns met min-score in best row")

    keep = keep[:a.max_zones]
    json.dump({"video": a.video, "frame": a.frame,
               "plate_mm": a.plate_mm,
               "plates": [[[float(x), float(y)] for x, y in c[1]]
                          for c in keep],
               "auto": True,
               "scores": [round(float(c[0]), 3) for c in keep],
               "info": [{k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in c[2].items()} for c in keep]},
              open(a.out, "w"), indent=2)

    vis = frame.copy()
    for i, c in enumerate(keep, 1):
        cv2.polylines(vis, [np.array(c[1], np.int32)], True, (0, 220, 0), 3)
        cv2.putText(vis, f"Z{i} {c[2]['slot_len_mm']:.0f}mm slots  "
                    f"score {c[0]:.2f}",
                    (c[1][0][0] + 6, c[1][0][1] + 28), 0, .7, (0, 0, 0), 4)
        cv2.putText(vis, f"Z{i} {c[2]['slot_len_mm']:.0f}mm slots  "
                    f"score {c[0]:.2f}",
                    (c[1][0][0] + 6, c[1][0][1] + 28), 0, .7, (0, 220, 0), 1)
    cv2.imwrite("autozones.png", vis)

    ws = [c[2]["mean_width_mm"] for c in keep]
    print(f"\nzones kept       : {len(keep)}")
    print(f"mean widths      : " + "  ".join(f"{w:.2f}" for w in ws))
    if len(ws) > 1:
        print(f"spread           : {max(ws)-min(ws):.2f} mm")
    print(f"\nwrote {a.out} and autozones.png")
    print("CHECK autozones.png, then:")
    print(f"  python video_gauge.py {a.video} --zones {a.out}")


if __name__ == "__main__":
    main()
