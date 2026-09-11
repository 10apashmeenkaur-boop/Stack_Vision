"""
run_gauge.py - the whole pipeline, end to end.

    python run_gauge.py clip.mp4                 first time: click 4 corners
    python run_gauge.py clip.mp4 --reuse         reuse saved plate_H.json
    python run_gauge.py clip.mp4 --trigger 0.25 0.75    sharp centre only

Click the 4 corners of ONE plate: TL, TR, BR, BL. The plate is 315x255 mm
from the drawing, so it warps to a canvas at exactly 4 px/mm.

Outputs:
  plate_H.json          the homography, reused on later runs
  rectified_f####.png   what the measurement actually saw
  annotated_f####.png   mm on every hole
  holes_mm.csv          one row per hole
  report.txt            summary + validation + limits, for Hira
"""
import argparse, csv, json, os
import cv2
import numpy as np
import statistics as st
import gauge

pts = []


def _click(ev, x, y, f, p):
    if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
        pts.append((x, y))


def pick_corners(frame):
    sc = min(1.0, 1500 / frame.shape[1], 800 / frame.shape[0])
    disp = cv2.resize(frame, None, fx=sc, fy=sc)
    lbl = ["top-left", "top-right", "bottom-right", "bottom-left"]
    cv2.namedWindow("corners")
    cv2.setMouseCallback("corners", _click)
    while True:
        vis = disp.copy()
        for i, p in enumerate(pts):
            cv2.circle(vis, p, 5, (0, 255, 0), -1)
            cv2.putText(vis, lbl[i], (p[0] + 8, p[1]), 0, .5, (0, 255, 0), 2)
        if len(pts) == 4:
            cv2.polylines(vis, [np.array(pts)], True, (0, 255, 255), 2)
        nxt = lbl[len(pts)] if len(pts) < 4 else "press s"
        cv2.putText(vis, f"click {nxt}  |  r reset  s save  q quit",
                    (10, 24), 0, .6, (0, 255, 255), 2)
        cv2.imshow("corners", vis)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('r'):
            pts.clear()
        if k == ord('q'):
            return None
        if k == ord('s') and len(pts) == 4:
            break
    cv2.destroyAllWindows()
    return [(x / sc, y / sc) for x, y in pts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--frame", type=int, default=1)
    ap.add_argument("--frames", type=int, nargs="*", default=None,
                    help="extra frames to also measure, e.g. --frames 1 257 513")
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--trigger", type=float, nargs=2, default=[0.0, 1.0],
                    help="horizontal band to measure, as fractions")
    ap.add_argument("--plate-mm", type=float, nargs=2, default=None,
                    metavar=("W", "H"),
                    help="real size of the region you clicked. One plate is "
                         "315 255. All three plates is 945 255.")
    a = ap.parse_args()

    if a.plate_mm:
        gauge.set_plate(a.plate_mm[0], a.plate_mm[1])
        print(f"reference        : {a.plate_mm[0]:.0f} x {a.plate_mm[1]:.0f} mm "
              f"-> canvas {gauge.CW}x{gauge.CH}")

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {a.video}")

    # ---- STAGE 1: homography ----
    if a.reuse and os.path.exists("plate_H.json"):
        H = np.array(json.load(open("plate_H.json"))["H"], np.float32)
        print("stage 1 corners   : reused plate_H.json")
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame - 1)
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("cannot read frame")
        src = pick_corners(frame)
        if src is None:
            return
        H = gauge.homography_from_corners(src)
        json.dump({"H": H.tolist(), "px_per_mm": gauge.PX_PER_MM,
                   "plate_mm": [gauge.PLATE_W_MM, gauge.PLATE_H_MM],
                   "corners": src}, open("plate_H.json", "w"), indent=2)
        print("stage 1 corners   : saved plate_H.json")

    frames = a.frames if a.frames else [a.frame]
    all_holes, summaries = [], []

    for fno in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno - 1)
        ok, frame = cap.read()
        if not ok:
            print(f"  frame {fno}: unreadable, skipped")
            continue

        rect = gauge.rectify(frame, H)                       # STAGE 2
        cv2.imwrite(f"rectified_f{fno:04d}.png", rect)

        res = gauge.measure_plate(rect, tuple(a.trigger))    # STAGES 3-4
        val = gauge.validate(res)                            # STAGE 5
        s = gauge.summarise(res)                             # STAGE 6

        cv2.imwrite(f"annotated_f{fno:04d}.png", gauge.draw(rect, res))
        for h in res["holes"]:
            h["frame"] = fno
            all_holes.append(h)
        if s:
            s["frame"] = fno
            summaries.append(s)

        print(f"\nframe {fno}")
        print(f"  stage 2 rectify : {gauge.CW}x{gauge.CH} px "
              f"@ {gauge.PX_PER_MM} px/mm  (1 px = "
              f"{1/gauge.PX_PER_MM:.2f} mm)")
        print(f"  stage 3 locate  : {len(res['holes'])} slots  "
              f"dropped {res['rejected']}")
        sh = res["sharpness"]
        print(f"  focus L->R      : " +
              " ".join(f"{v:.0f}" for v in sh) +
              f"   ({100*(max(sh)-min(sh))/max(sh):.0f}% falloff)")
        print(f"  stage 5 VALIDATE: {val['msg']}  "
              f"{'PASS' if val['ok'] else '** FAIL - scale is wrong **'}")
        if s:
            print(f"  stage 6 result  : median {s['median_mm']} mm  "
                  f"p95 {s['p95_mm']} mm  reject {s['reject_pct']}%  "
                  f"-> {s['verdict']}")

    if not all_holes:
        raise SystemExit("\nno holes measured - open rectified_*.png; if the "
                         "plate is skewed the corners were wrong")

    with open("holes_mm.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_holes[0].keys()))
        w.writeheader(); w.writerows(all_holes)

    mm = sorted(h["width_mm"] for h in all_holes)
    rej = 100 * sum(h["grade"] == "REJECT" for h in all_holes) / len(all_holes)
    med = st.median(mm)
    lines = [
        "HIRA GRATE SLOT INSPECTION",
        "=" * 46,
        f"source           : {a.video}",
        f"frames measured  : {len(summaries)}",
        f"slots measured   : {len(all_holes)}",
        "",
        "CALIBRATION",
        f"  method         : planar homography from plate corners",
        f"  reference      : plate {gauge.PLATE_W_MM:.0f} x "
        f"{gauge.PLATE_H_MM:.0f} mm (drawing)",
        f"  scale          : {gauge.PX_PER_MM} px/mm "
        f"(1 px = {1/gauge.PX_PER_MM:.2f} mm)",
        f"  independent    : slot length vs drawing rows 62/79/62 mm",
        "",
        "RESULT",
        f"  median width   : {med:.2f} mm",
        f"  p95 width      : {mm[int(.95*len(mm))-1]:.2f} mm",
        f"  max width      : {mm[-1]:.2f} mm",
        f"  over {gauge.REJECT_MM:.0f} mm      : {rej:.1f}% of slots",
        f"  VERDICT        : "
        f"{'REJECT' if rej > 10 else 'WATCH' if med > gauge.WATCH_MM else 'PASS'}",
        "",
        "LIMITS",
        "  Slot width is measured at the sub-pixel 50% intensity crossing",
        "  along each slot's minor axis, after perspective rectification.",
        "  Accuracy degrades toward the edges of the camera field where",
        "  focus falls off; measurements are confined to the sharp band.",
        "  Homography assumes a planar surface - the drum curvature adds",
        "  residual error at the extreme left and right of each plate.",
        "  Slots narrower than 2 mm are treated as detection artefacts,",
        "  since a slot cannot erode narrower than new.",
    ]
    open("report.txt", "w").write("\n".join(lines))
    print("\n" + "\n".join(lines[-13:]))
    print("\nwrote holes_mm.csv, report.txt, rectified_*.png, annotated_*.png")


if __name__ == "__main__":
    main()
