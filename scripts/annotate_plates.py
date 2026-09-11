"""
annotate_plates.py - click the corners of every plate you want to measure.

This is the plate equivalent of your CVAT hole annotation. You do it ONCE
per camera; the camera is fixed so the corners never move.

    python annotate_plates.py clip.mp4

For each plate click 4 corners: top-left, top-right, bottom-right,
bottom-left. Then press SPACE to start the next plate. Press S when done.

Keys
  click   place a corner
  space   finish this plate, start the next
  u       undo last corner
  d       delete the last completed plate
  s       save all plates -> plates.json
  q       quit without saving

Then measure them all:
    python measure_all.py clip.mp4
"""
import argparse, json
import cv2
import numpy as np

cur, done = [], []


def _click(ev, x, y, flags, p):
    if ev == cv2.EVENT_LBUTTONDOWN and len(cur) < 4:
        cur.append((x, y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--frame", type=int, default=100)
    ap.add_argument("--out", default="plates.json")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame - 1)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("cannot read frame")

    sc = min(1.0, 1500 / frame.shape[1], 820 / frame.shape[0])
    disp = cv2.resize(frame, None, fx=sc, fy=sc)
    lbl = ["top-left", "top-right", "bottom-right", "bottom-left"]
    cols = [(0, 220, 0), (0, 200, 255), (255, 160, 0), (255, 0, 255),
            (0, 255, 255), (200, 0, 200)]

    cv2.namedWindow("annotate plates")
    cv2.setMouseCallback("annotate plates", _click)

    while True:
        vis = disp.copy()
        for i, p in enumerate(done):
            c = cols[i % len(cols)]
            cv2.polylines(vis, [np.array(p, np.int32)], True, c, 2)
            cv2.putText(vis, f"P{i+1}", (p[0][0] + 6, p[0][1] + 22),
                        0, .7, c, 2)
        for i, p in enumerate(cur):
            cv2.circle(vis, p, 5, (0, 255, 0), -1)
            cv2.putText(vis, lbl[i], (p[0] + 8, p[1]), 0, .45, (0, 255, 0), 1)
        if len(cur) == 4:
            cv2.polylines(vis, [np.array(cur, np.int32)], True,
                          (0, 255, 255), 2)

        nxt = lbl[len(cur)] if len(cur) < 4 else "SPACE for next plate"
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 30), (25, 25, 25), -1)
        cv2.putText(vis, f"plate {len(done)+1}: click {nxt}   |   "
                    f"{len(done)} saved   |   u undo  d del  s save  q quit",
                    (8, 20), 0, .55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("annotate plates", vis)

        k = cv2.waitKey(20) & 0xFF
        if k == ord('u') and cur:
            cur.pop()
        elif k == ord('d') and done:
            done.pop()
        elif k == ord(' ') and len(cur) == 4:
            done.append(list(cur)); cur.clear()
        elif k == ord('q'):
            return
        elif k == ord('s'):
            if len(cur) == 4:
                done.append(list(cur)); cur.clear()
            break
    cv2.destroyAllWindows()

    if not done:
        raise SystemExit("no plates annotated")

    plates = [[[x / sc, y / sc] for x, y in p] for p in done]
    json.dump({"video": a.video, "frame": a.frame,
               "plate_mm": [315.0, 255.0], "plates": plates},
              open(a.out, "w"), indent=2)
    print(f"saved {len(plates)} plates -> {a.out}")
    print("\nnext:  python measure_all.py " + a.video)


if __name__ == "__main__":
    main()
