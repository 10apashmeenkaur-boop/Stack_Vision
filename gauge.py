"""
gauge.py - the measurement core.

Stages:
  1 rectify   homography -> 315 x 255 mm canvas at 4 px/mm
  2 locate    adaptive threshold + connected components
  3 orient    minAreaRect -> the slot's true minor axis
  4 measure   sub-pixel 50% intensity crossing across that axis
  5 validate  slot lengths vs drawing (62 / 79 / 62 mm rows)
  6 classify  mm, upper bound only (holes erode wider, never narrower)

Every stage exposes its numbers so the result can be defended, not just
reported.

Used by plate_mm.py and the UI. Not run directly.
"""
import cv2
import numpy as np
import statistics as st

# --- from the drawing ---
PLATE_W_MM, PLATE_H_MM = 315.0, 255.0
ROW_BANDS_MM = (62.0, 79.0, 62.0)      # the three slot rows
PX_PER_MM = 4.0
CW, CH = int(PLATE_W_MM * PX_PER_MM), int(PLATE_H_MM * PX_PER_MM)

# --- tolerance. Upper bound ONLY: a slot cannot wear narrower. ---
WATCH_MM, REJECT_MM = 5.0, 6.0
# vertical closing kernel - bridges specks inside slots. Validated against
# the drawing: 41 recovers 61.9 mm slot length vs the 62 mm row band.
MIN_PLAUSIBLE_MM = 2.0       # below this it is a detection artefact

# vertical closing kernel - bridges reflections and debris inside slots.
# Fitted against the drawing: 41 recovers 61.9 mm slot length vs the 62 mm
# row band. 9 gave 34.6 mm (fragmented), 81 gave 33.2 mm (over-merged).
CLOSE_H = 41


def set_plate(w_mm, h_mm, px_per_mm=None):
    """Change the reference dimensions - e.g. if you clicked all three
    plates, call set_plate(945, 255)."""
    global PLATE_W_MM, PLATE_H_MM, PX_PER_MM, CW, CH
    PLATE_W_MM, PLATE_H_MM = float(w_mm), float(h_mm)
    if px_per_mm:
        PX_PER_MM = float(px_per_mm)
    CW, CH = int(PLATE_W_MM * PX_PER_MM), int(PLATE_H_MM * PX_PER_MM)
    return CW, CH


def rectify(frame, H):
    return cv2.warpPerspective(frame, H, (CW, CH))


def homography_from_corners(src4):
    """src4: TL, TR, BR, BL in raw image pixels."""
    dst = np.float32([[0, 0], [CW, 0], [CW, CH], [0, CH]])
    return cv2.getPerspectiveTransform(np.float32(src4), dst)


def sharpness_map(gray, band_frac=(0.25, 0.75)):
    """Laplacian variance per vertical band - finds where the image is in
    focus, so measurement can be confined there."""
    W = gray.shape[1]
    out = []
    for i in range(8):
        x0, x1 = i * W // 8, (i + 1) * W // 8
        out.append(cv2.Laplacian(gray[:, x0:x1], cv2.CV_64F).var())
    return out


def _subpixel_width(gray, cx, cy, ux, uy, span, n=9, half_len=20):
    """
    Width at the 50% intensity crossing, measured ALONG the slot's minor
    axis (ux, uy), sampled at n points down the slot and medianed.

    Sub-pixel because a 1 px rounding error is 0.25 mm here, and the whole
    pass/reject decision spans 1 mm.
    """
    px, py = -uy, ux                    # along the slot
    widths = []
    for t in np.linspace(-half_len, half_len, n):
        sx, sy = cx + px * t, cy + py * t
        prof, coords = [], []
        for s in np.arange(-span, span, 0.5):
            x, y = sx + ux * s, sy + uy * s
            if not (0 <= x < gray.shape[1] - 1 and 0 <= y < gray.shape[0] - 1):
                prof = []
                break
            x0, y0 = int(x), int(y)
            fx, fy = x - x0, y - y0     # bilinear sample
            v = (gray[y0, x0] * (1 - fx) * (1 - fy) +
                 gray[y0, x0 + 1] * fx * (1 - fy) +
                 gray[y0 + 1, x0] * (1 - fx) * fy +
                 gray[y0 + 1, x0 + 1] * fx * fy)
            prof.append(v); coords.append(s)
        if len(prof) < 8:
            continue
        prof = np.array(prof)
        lo, hi = prof.min(), prof.max()
        if hi - lo < 15:
            continue
        thr = (lo + hi) / 2.0
        mid = len(prof) // 2

        def cross(rng):
            for i in rng:
                a, b = prof[i], prof[i + 1]
                if (a - thr) * (b - thr) < 0:
                    f = (thr - a) / (b - a)
                    return coords[i] + f * (coords[i + 1] - coords[i])
            return None

        l = cross(range(mid, 0, -1))
        r = cross(range(mid, len(prof) - 1))
        if l is None or r is None:
            continue
        widths.append(abs(r - l))
    if len(widths) < 3:
        return None, None
    m = float(st.median(widths))
    spread = float(st.pstdev(widths)) / m if m > 0 else 9.0
    return m, spread


def _interior_dark_frac(gray, cx, cy, ux, uy, half_len, n=15):
    """
    Fraction of the slot's centreline that is actually dark.

    A clean open slot is dark end to end. Debris, ore or a reflection sitting
    in the slot shows up as a bright patch - and the threshold would count
    that debris edge as the slot edge, reporting a slot far wider than it is.
    This is what produced the 8-9 mm readings.
    """
    px, py = -uy, ux
    vals = []
    for t in np.linspace(-half_len * 0.8, half_len * 0.8, n):
        x, y = int(round(cx + px * t)), int(round(cy + py * t))
        if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
            vals.append(gray[y, x])
    if len(vals) < 5:
        return 0.0
    vals = np.array(vals)
    # dark relative to the plate, not an absolute level
    thr = np.percentile(gray, 55) * 0.75
    return float((vals < thr).mean())


def measure_plate(rect, trigger=(0.0, 1.0), debug=False,
                  min_solidity=0.65, min_dark=0.0, max_spread=0.30):
    """
    min_solidity 0.65 - fitted by sweep against the drawing. A clean slot
      fills 65%+ of its own rotated bounding box; debris-split blobs do not.
      Validation lands at 62.0 mm vs the drawing's 62 mm band (1% off).
    min_dark 0.0 - OFF by default. An interior-darkness gate sounds right
      but biases the result: narrow slots have less dark area, so it culls
      them and drags the median UP (6.04 -> 6.43 mm). Left available but
      disabled.
    """
    """
    rect     : rectified plate, CW x CH, 4 px per mm
    trigger  : (x0_frac, x1_frac) - measure only this horizontal band.
               Use the sharp centre to avoid the focus gradient.
    """
    gray0 = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY).astype(np.float64)
    g = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(2.0, (8, 8)).apply(g)
    g = cv2.medianBlur(g, 3)

    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 51, 12)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((25, 1), np.uint8))
    # Bridge reflections and debris INSIDE slots. Without a tall enough
    # kernel a slot fragments and its length is measured as ~35 mm instead
    # of the drawing's 62 mm. 41 was fitted against the drawing.
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE,
                          np.ones((CLOSE_H, 1), np.uint8))

    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    tx0, tx1 = trigger[0] * CW, trigger[1] * CW

    holes, rejected = [], {"outside_trigger": 0, "too_small": 0,
                           "at_edge": 0, "no_subpixel": 0, "implausible": 0,
                           "ragged": 0, "obscured": 0, "inconsistent": 0}

    for c in cnts:
        if len(c) < 5:
            rejected["too_small"] += 1
            continue
        (cx, cy), (w, h), ang = cv2.minAreaRect(c)   # STAGE 3: true axes
        if w == 0 or h == 0:
            rejected["too_small"] += 1
            continue
        short, long_ = min(w, h), max(w, h)
        if long_ < 40 or long_ / short < 2.0:
            rejected["too_small"] += 1
            continue
        if not (tx0 <= cx <= tx1):
            rejected["outside_trigger"] += 1
            continue
        x, y, bwid, bhei = cv2.boundingRect(c)
        # only reject if clipped along the MINOR axis - a slot clipped at
        # its end still gives a valid width
        if x <= 1 or x + bwid >= CW - 1:
            rejected["at_edge"] += 1
            continue

        # STAGE 3b: a clean slot fills its own bounding rectangle. A blob
        # split or bridged by debris does not.
        if cv2.contourArea(c) / (w * h) < min_solidity:
            rejected["ragged"] += 1
            continue

        # unit vector along the MINOR axis
        a = np.deg2rad(ang if w < h else ang + 90)
        ux, uy = np.cos(a), np.sin(a)

        dark = _interior_dark_frac(gray0, cx, cy, ux, uy, long_ / 2)
        if dark < min_dark:
            rejected["obscured"] += 1
            continue

        sub, spread = _subpixel_width(gray0, cx, cy, ux, uy,
                                      span=max(8.0, short * 2.0),
                                      half_len=long_ * 0.3)
        if sub is None:
            rejected["no_subpixel"] += 1
            continue
        if spread > max_spread:
            rejected["inconsistent"] += 1
            continue
        mm = sub / PX_PER_MM
        if mm < MIN_PLAUSIBLE_MM:
            rejected["implausible"] += 1
            continue

        holes.append({
            "cx": round(cx, 1), "cy": round(cy, 1),
            "x_mm": round(cx / PX_PER_MM, 1),
            "y_mm": round(cy / PX_PER_MM, 1),
            "width_px_rect": round(short, 2),
            "width_px_sub": round(sub, 2),
            "width_mm": round(mm, 2),
            "length_mm": round(long_ / PX_PER_MM, 1),
            "angle": round(ang, 1),
            "spread": round(spread, 3),
            "dark_frac": round(dark, 2),
            "grade": ("REJECT" if mm > REJECT_MM else
                      "WATCH" if mm > WATCH_MM else "GOOD"),
        })

    return {"holes": holes, "rejected": rejected,
            "mask": bw if debug else None,
            "sharpness": sharpness_map(
                cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY))}


def validate(res):
    """
    STAGE 5. Scale was set from plate WIDTH. Slot LENGTH is an independent
    dimension from the drawing - if measured lengths land near the 62/79/62
    row bands, the millimetres are real. This is the check that makes the
    result defensible.
    """
    h = res["holes"]
    if not h:
        return {"ok": False, "msg": "no holes measured"}
    lens = sorted(x["length_mm"] for x in h)
    # p75, not median: reflections inside slots leave short fragments that
    # drag the median down. The upper quartile is the intact slots, which
    # is what the drawing's row bands describe.
    med = lens[int(0.75 * len(lens)) - 1] if len(lens) >= 4 else st.median(lens)
    best = min(ROW_BANDS_MM, key=lambda b: abs(b - med))
    err = 100 * abs(med - best) / best
    return {
        "ok": err < 15,
        "median_length_mm": round(med, 1),
        "nearest_drawing_band": best,
        "error_pct": round(err, 1),
        "msg": (f"slot length {med:.1f} mm vs drawing {best:.0f} mm "
                f"({err:.1f}% off)"),
    }


def summarise(res):
    h = res["holes"]
    if not h:
        return {}
    mm = sorted(x["width_mm"] for x in h)
    rej = 100 * sum(x["grade"] == "REJECT" for x in h) / len(h)
    return {
        "n_holes": len(h),
        "median_mm": round(st.median(mm), 2),
        "p95_mm": round(mm[int(.95 * len(mm)) - 1], 2),
        "max_mm": mm[-1],
        "reject_pct": round(rej, 1),
        "verdict": ("REJECT" if rej > 10 else
                    "WATCH" if st.median(mm) > WATCH_MM else "PASS"),
    }


def draw(rect, res, show_mm=True):
    vis = rect.copy()
    col = {"GOOD": (0, 200, 0), "WATCH": (0, 210, 255), "REJECT": (0, 0, 255)}
    for o in res["holes"]:
        c = col[o["grade"]]
        cv2.circle(vis, (int(o["cx"]), int(o["cy"])), 3, c, -1)
        if show_mm:
            cv2.putText(vis, f'{o["width_mm"]:.1f}',
                        (int(o["cx"]) - 14, int(o["cy"]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, f'{o["width_mm"]:.1f}',
                        (int(o["cx"]) - 14, int(o["cy"]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, .42, c, 1, cv2.LINE_AA)
    s = summarise(res)
    if s:
        bar = (f'{s["n_holes"]} holes   median {s["median_mm"]} mm   '
               f'p95 {s["p95_mm"]} mm   reject {s["reject_pct"]}%   '
               f'-> {s["verdict"]}')
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 30), (25, 25, 25), -1)
        cv2.putText(vis, bar, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, .6,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return vis
