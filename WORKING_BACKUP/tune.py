"""
STEP 1 - tune the detector against YOUR CVAT annotations.

Your 378 hand-corrected boxes are ground truth. This sweeps the detector's
parameters, matches its boxes to yours by overlap, and reports which
settings reproduce your pixel widths. Prints the winning flags at the end.

    python tune.py annotations.xml roi_01.png roi_02.png roi_03.png

Needs auto_boxes.py in the same folder.
"""
import argparse, itertools, sys
import xml.etree.ElementTree as ET
import numpy as np
from auto_boxes import detect


class NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def load_gt(xml_path):
    root = ET.parse(xml_path).getroot()
    gt = {}
    for img in root.iter("image"):
        boxes = []
        for b in img.iter("box"):
            x1, y1 = float(b.get("xtl")), float(b.get("ytl"))
            x2, y2 = float(b.get("xbr")), float(b.get("ybr"))
            boxes.append((x1, y1, x2, y2))
        gt[img.get("name")] = boxes
    return gt


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua


def score(gt_boxes, det_boxes, thr=0.30):
    """Greedy match. Returns recall, precision, median |width error| in px."""
    if not det_boxes:
        return 0.0, 0.0, 99.0
    used = set()
    errs = []
    for g in gt_boxes:
        best, bi = 0.0, -1
        for i, d in enumerate(det_boxes):
            if i in used:
                continue
            v = iou(g, d)
            if v > best:
                best, bi = v, i
        if best >= thr:
            used.add(bi)
            gw = g[2] - g[0]
            dw = det_boxes[bi][2] - det_boxes[bi][0]
            errs.append(abs(gw - dw))
    rec = len(used) / len(gt_boxes)
    prec = len(used) / len(det_boxes)
    return rec, prec, (float(np.median(errs)) if errs else 99.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--iou", type=float, default=0.30)
    a = ap.parse_args()

    gt = load_gt(a.xml)
    imgs = [p for p in a.images if p.split("\\")[-1].split("/")[-1] in gt]
    if not imgs:
        sys.exit(f"image names in xml {list(gt)} don't match the files you passed")

    grid = list(itertools.product(
        [11, 15, 21, 25],          # open_h
        [31, 51, 71],              # block
        [4, 8, 12],                # C
    ))

    print(f"ground truth: {sum(len(v) for v in gt.values())} boxes")
    print(f"sweeping {len(grid)} combos over {len(imgs)} images\n")
    print(f"{'open_h':>6} {'block':>6} {'C':>3} | {'recall':>6} {'prec':>6} "
          f"{'w_err_px':>8} {'score':>6}")
    print("-" * 52)

    results = []
    for open_h, block, C in grid:
        ns = NS(open_h=open_h, block=block, C=C, min_len=12,
                min_aspect=1.8, solidity=0.30, band=[0.35, 3.0], debug=False)
        recs, precs, errs = [], [], []
        for p in imgs:
            name = p.split("\\")[-1].split("/")[-1]
            _, cand, _, _, _ = detect(p, ns)
            det = [(x, y, x + w, y + h) for x, y, w, h in cand]
            r, pr, e = score(gt[name], det, a.iou)
            recs.append(r); precs.append(pr); errs.append(e)
        R, P, E = np.mean(recs), np.mean(precs), np.median(errs)
        # want high recall, high precision, low width error
        s = R * P / (1.0 + E)
        results.append((s, open_h, block, C, R, P, E))
        print(f"{open_h:>6} {block:>6} {C:>3} | {R:>6.2f} {P:>6.2f} "
              f"{E:>8.2f} {s:>6.3f}")

    results.sort(reverse=True)
    s, open_h, block, C, R, P, E = results[0]
    print("\n" + "=" * 52)
    print(f"BEST: recall {R:.2f}  precision {P:.2f}  median width error {E:.2f} px")
    print("\nuse these flags everywhere from now on:")
    print(f"  --open-h {open_h} --block {block} --C {C} "
          f"--min-len 12 --min-aspect 1.8 --solidity 0.30 --band 0.35 3.0")
    if E > 2.0:
        print(f"\nWARNING: {E:.1f} px median width error against your own boxes.")
        print("On 11 px holes that is a large fraction. The detector cannot")
        print("match your annotation at this resolution.")


if __name__ == "__main__":
    main()
