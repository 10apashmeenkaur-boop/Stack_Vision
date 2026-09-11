"""
Run the tuned detector over the whole video, inside the ROI only.

    python run_video.py clip.mp4

Uses the settings tune.py found (0.00 px width error vs your CVAT boxes).
Crop is baked in from pick_crop.py.

Outputs:
  output.mp4        annotated ROI, every frame, holes boxed + width in px
  frames.csv        per frame: hole count, median/min/max width px
  holes_video.csv   every hole in every frame

Keys while the preview window is open:  q = quit early,  space = pause
"""
import argparse, csv
import cv2
import numpy as np

# --- from pick_crop.py ---
CROP_W, CROP_H, CROP_X, CROP_Y = 1114, 372, 402, 151

# --- from tune.py: recall 0.81, precision 0.84, width error 0.00 px ---
OPEN_H, BLOCK, C = 15, 51, 12
MIN_LEN, MIN_ASPECT, SOLIDITY = 12, 1.8, 0.30
BAND_LO, BAND_HI = 0.35, 3.0


def _profile(g, axis):
    p = g.mean(axis=axis).astype(float)
    return (p - p.mean()) / (p.std() + 1e-9)


def _shift(a, b, rng=25):
    """Best 1-D alignment of two profiles. Robust where phaseCorrelate is
    not: the plate pattern is periodic, so 2-D phase correlation locks onto
    the pattern period or onto noise instead of true belt motion."""
    best, bc = 0, -9.0
    for s in range(-rng, rng + 1):
        if s >= 0:
            x, y = a[s:], b[:len(b) - s]
        else:
            x, y = a[:len(a) + s], b[-s:]
        if len(x) < 50:
            continue
        c = float(np.dot(x, y) / len(x))
        if c > bc:
            best, bc = s, c
    return best, bc


def detect(roi):
    H, W = roi.shape[:2]
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(2.0, (8, 8)).apply(g)
    g = cv2.medianBlur(g, 3)

    blk = BLOCK if BLOCK % 2 else BLOCK + 1
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, blk, C)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((OPEN_H, 1), np.uint8))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE,
                          np.ones((max(3, OPEN_H // 2), 1), np.uint8))

    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    cand = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h < MIN_LEN:
            continue
        if h / max(w, 1) < MIN_ASPECT:
            continue
        if area / float(w * h) < SOLIDITY:
            continue
        if x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
            continue
        if h > 0.9 * H:
            continue
        cand.append((x, y, w, h))

    if len(cand) >= 8:
        med = float(np.median([c[2] for c in cand]))
        cand = [c for c in cand if BAND_LO * med <= c[2] <= BAND_HI * med]
    return cand


def draw(roi, cand, frame_idx):
    vis = roi.copy()
    if cand:
        med = float(np.median([c[2] for c in cand]))
        for (x, y, w, h) in cand:
            r = w / med
            col = (0, 200, 0) if r < 1.2 else ((0, 210, 255) if r < 1.5
                                               else (0, 0, 255))
            cv2.rectangle(vis, (x, y), (x + w, y + h), col, 2)
            cv2.putText(vis, str(w), (x - 2, max(11, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, .38, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, str(w), (x - 2, max(11, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, .38, col, 1, cv2.LINE_AA)
        bar = f"frame {frame_idx}   {len(cand)} holes   median {med:.0f}px"
    else:
        bar = f"frame {frame_idx}   no holes"
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 26), (25, 25, 25), -1)
    cv2.putText(vis, bar, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--every", type=int, default=1, help="process every Nth frame")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{a.video}: {total} frames @ {fps:.1f} fps")
    print(f"ROI {CROP_W}x{CROP_H} at ({CROP_X},{CROP_Y})")

    out = cv2.VideoWriter("output.mp4",
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          fps / a.every, (CROP_W, CROP_H))

    fr = open("frames.csv", "w", newline="")
    fw = csv.writer(fr)
    fw.writerow(["frame", "n_holes", "median_px", "min_px", "max_px",
                 "dx", "dy"])
    hr = open("holes_video.csv", "w", newline="")
    hw = csv.writer(hr)
    hw.writerow(["frame", "cx", "cy", "width_px", "length_px"])

    idx, done, paused = 0, 0, False
    prev_g = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if (idx - 1) % a.every:
            continue

        roi = frame[CROP_Y:CROP_Y + CROP_H, CROP_X:CROP_X + CROP_W]
        if roi.shape[0] != CROP_H or roi.shape[1] != CROP_W:
            print(f"frame {idx}: crop outside frame bounds - check CROP_*")
            break

        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        cur_g = (_profile(g, 1), _profile(g, 0))     # (row prof, col prof)
        if prev_g is not None:
            dy, _ = _shift(prev_g[0], cur_g[0])
            dx, _ = _shift(prev_g[1], cur_g[1])
        else:
            dx = dy = 0.0
        prev_g = cur_g

        cand = detect(roi)
        vis = draw(roi, cand, idx)
        out.write(vis)
        done += 1

        if cand:
            ws = [c[2] for c in cand]
            fw.writerow([idx, len(cand), int(np.median(ws)), min(ws),
                         max(ws), round(dx, 3), round(dy, 3)])
            for (x, y, w, h) in cand:
                hw.writerow([idx, x + w // 2, y + h // 2, w, h])
        else:
            fw.writerow([idx, 0, "", "", "", round(dx, 3), round(dy, 3)])

        if not a.no_window:
            cv2.imshow("hira - q quit, space pause", vis)
            k = cv2.waitKey(1 if not paused else 0) & 0xFF
            if k == ord('q'):
                break
            if k == ord(' '):
                paused = not paused

        if done % 100 == 0:
            print(f"  {idx}/{total}")

    cap.release(); out.release(); fr.close(); hr.close()
    cv2.destroyAllWindows()

    import statistics as st
    rows = list(csv.DictReader(open("frames.csv")))
    counts = [int(r["n_holes"]) for r in rows if r["n_holes"]]
    meds = [int(r["median_px"]) for r in rows if r["median_px"]]
    print(f"\nprocessed        : {done} frames")
    if counts:
        print(f"holes per frame  : median {st.median(counts):.0f}  "
              f"range {min(counts)}-{max(counts)}")
    if meds:
        print(f"width px         : median {st.median(meds):.0f}  "
              f"range {min(meds)}-{max(meds)}")
        print(f"stability        : stdev {st.pstdev(meds):.2f} px across frames")
    print("\nwrote output.mp4, frames.csv, holes_video.csv")


if __name__ == "__main__":
    main()
