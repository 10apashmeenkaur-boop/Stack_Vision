"""
Click the top-left then bottom-right of your ROI (the three front plates).
Prints the ffmpeg crop string and the exact command to cut 3 frames.

    python pick_crop.py clip.mp4

Keys:  r = reset   s = save+print   q = quit
"""
import cv2, sys

pts = []


def on_click(ev, x, y, flags, p):
    if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
        pts.append((x, y))


def main(path):
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not ok:
        sys.exit(f"cannot read {path}")

    H, W = frame.shape[:2]
    print(f"video: {W}x{H}, {n_frames} frames")

    # shrink for display if the frame is bigger than the screen
    scale = min(1.0, 1400 / W, 800 / H)
    disp = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame.copy()

    cv2.namedWindow("pick ROI")
    cv2.setMouseCallback("pick ROI", on_click)
    while True:
        vis = disp.copy()
        for p in pts:
            cv2.circle(vis, p, 5, (0, 255, 0), -1)
        if len(pts) == 2:
            cv2.rectangle(vis, pts[0], pts[1], (0, 255, 255), 2)
        cv2.putText(vis, "TL then BR   |   r reset   s save   q quit",
                    (10, 25), 0, 0.6, (0, 255, 255), 2)
        cv2.imshow("pick ROI", vis)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('r'):
            pts.clear()
        if k == ord('q'):
            return
        if k == ord('s') and len(pts) == 2:
            break
    cv2.destroyAllWindows()

    (x1, y1), (x2, y2) = pts
    x1, x2 = sorted((int(x1 / scale), int(x2 / scale)))
    y1, y2 = sorted((int(y1 / scale), int(y2 / scale)))
    w, h = x2 - x1, y2 - y1
    w -= w % 2
    h -= h % 2

    crop = f"crop={w}:{h}:{x1}:{y1}"
    a, b, c = 0, n_frames // 3, (2 * n_frames) // 3

    print("\n--- crop ---")
    print(crop)
    print("\n--- run this next ---")
    print(f'ffmpeg -i {path} -vf "{crop},'
          f"select='eq(n\\,{a})+eq(n\\,{b})+eq(n\\,{c})'\" -vsync 0 roi_%02d.png")
    print("\nthen upload roi_01.png roi_02.png roi_03.png to CVAT as an IMAGE task")

    cv2.imwrite("roi_check.png", frame[y1:y2, x1:x2])
    print("wrote roi_check.png - open it, confirm it is the three front plates")


if __name__ == "__main__":
    main(sys.argv[1])
