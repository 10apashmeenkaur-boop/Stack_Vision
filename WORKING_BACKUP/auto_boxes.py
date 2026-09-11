"""
v3 - readable previews + tells you which filter is eating your holes.

    python auto_boxes.py roi_01.png roi_02.png roi_03.png --debug

New:
  - every box labelled with its width in px
  - thick boxes, colour-coded by width vs median
      green  = near median      (normal)
      yellow = 1.2-1.5x median  (worn)
      red    = >1.5x median     (very worn / merged)
  - REJECT COUNTS printed: shows exactly which filter dropped how many
    blobs, so you know which flag to change instead of guessing
  - looser defaults so fewer holes are missed

Flags:
  --open-h 15    taller breaks merged slots harder (21, 25)
  --C 8          threshold offset. lower (4) finds fainter holes
  --min-len 20   minimum slot length px
  --solidity .35 lower finds more, allows scrappier blobs
  --band 0.40 2.5  width outlier band as fraction of median. widen to keep more
  --no-text      turn off the px labels
  --debug        writes mask_*.png
"""
import argparse, csv, os
import xml.etree.ElementTree as ET
import cv2
import numpy as np


def detect(path, a):
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"cannot read {path}")
    H, W = img.shape[:2]

    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(2.0, (8, 8)).apply(g)
    g = cv2.medianBlur(g, 3)

    blk = a.block if a.block % 2 else a.block + 1
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, blk, a.C)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((a.open_h, 1), np.uint8))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE,
                          np.ones((max(3, a.open_h // 2), 1), np.uint8))

    if a.debug:
        cv2.imwrite(f"mask_{os.path.basename(path)}", bw)

    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, 8)

    rej = {"too_short": 0, "not_elongated": 0, "hollow": 0,
           "at_edge": 0, "full_height": 0, "width_outlier": 0}
    cand = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h < a.min_len:
            rej["too_short"] += 1; continue
        if h / max(w, 1) < a.min_aspect:
            rej["not_elongated"] += 1; continue
        if area / float(w * h) < a.solidity:
            rej["hollow"] += 1; continue
        if x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
            rej["at_edge"] += 1; continue
        if h > 0.9 * H:
            rej["full_height"] += 1; continue
        cand.append((x, y, w, h))

    if len(cand) >= 8:
        med = float(np.median([c[2] for c in cand]))
        keep = [c for c in cand if a.band[0] * med <= c[2] <= a.band[1] * med]
        rej["width_outlier"] = len(cand) - len(keep)
        cand = keep

    return img, cand, W, H, rej


def preview(img, cand, out, show_text):
    vis = img.copy()
    if not cand:
        cv2.imwrite(out, vis); return
    med = float(np.median([c[2] for c in cand]))
    for (x, y, w, h) in cand:
        r = w / med
        col = (0, 200, 0) if r < 1.2 else ((0, 210, 255) if r < 1.5 else (0, 0, 255))
        cv2.rectangle(vis, (x, y), (x + w, y + h), col, 2)
        if show_text:
            cv2.putText(vis, str(w), (x - 2, max(11, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, str(w), (x - 2, max(11, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 26), (25, 25, 25), -1)
    cv2.putText(vis, f"{len(cand)} holes   median width {med:.0f}px",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                cv2.LINE_AA)
    cv2.imwrite(out, vis)


def write_xml(entries, out="auto_annotations.xml"):
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    for idx, (name, W, H, cand) in enumerate(entries):
        im = ET.SubElement(root, "image", id=str(idx), name=name,
                           width=str(W), height=str(H))
        for (x, y, w, h) in cand:
            ET.SubElement(im, "box", label="grate_hole", occluded="0",
                          source="auto", z_order="0",
                          xtl=f"{x:.2f}", ytl=f"{y:.2f}",
                          xbr=f"{x+w:.2f}", ybr=f"{y+h:.2f}")
    ET.ElementTree(root).write(out, encoding="utf-8", xml_declaration=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="+")
    p.add_argument("--open-h", type=int, default=15, dest="open_h")
    p.add_argument("--block", type=int, default=51)
    p.add_argument("--C", type=int, default=8)
    p.add_argument("--min-len", type=int, default=20, dest="min_len")
    p.add_argument("--min-aspect", type=float, default=2.2, dest="min_aspect")
    p.add_argument("--solidity", type=float, default=0.35)
    p.add_argument("--band", type=float, nargs=2, default=[0.40, 2.5])
    p.add_argument("--no-text", action="store_true")
    p.add_argument("--debug", action="store_true")
    a = p.parse_args()

    entries, rows, total_rej = [], [], {}
    for path in a.images:
        name = os.path.basename(path)
        img, cand, W, H, rej = detect(path, a)
        entries.append((name, W, H, cand))
        preview(img, cand, f"preview_{name}", not a.no_text)
        for (x, y, w, h) in cand:
            rows.append({"image": name, "cx": x + w // 2, "cy": y + h // 2,
                         "width_px": w, "length_px": h})
        for k, v in rej.items():
            total_rej[k] = total_rej.get(k, 0) + v
        print(f"{name}: {len(cand)} holes -> preview_{name}")

    print("\nblobs rejected by filter:")
    for k, v in sorted(total_rej.items(), key=lambda kv: -kv[1]):
        if v:
            print(f"  {k:<15} {v}")
    print("  ^ if a big number here is eating real holes, loosen that filter:")
    print("    too_short->--min-len 12  not_elongated->--min-aspect 1.5")
    print("    hollow->--solidity 0.25  width_outlier->--band 0.3 3.0")

    if not rows:
        raise SystemExit("\nnothing kept. try --C 4 --min-len 12")

    xml = write_xml(entries)
    with open("auto_holes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    wd = sorted(r["width_px"] for r in rows)
    print(f"\ntotal        : {len(rows)}")
    print(f"width px med : {wd[len(wd)//2]}   range {wd[0]}-{wd[-1]}")
    print(f"wrote {xml} + auto_holes.csv")


if __name__ == "__main__":
    main()
