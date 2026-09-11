"""
Turn a CVAT export into per-hole pixel sizes and mm.

    python cvat_to_px.py annotations.xml
    python cvat_to_px.py annotations.xml --pitch-mm 12.0
    python cvat_to_px.py annotations.xml --mm-per-px 0.1205

Export from CVAT as: Menu -> Export annotations -> "CVAT for images 1.1"
Unzip it; the file you want is annotations.xml

Scale:
  --pitch-mm    BEST. Uses centre-to-centre hole spacing as the ruler.
                Pitch is manufactured, it does not wear. Give the number
                from Hira's drawing.
  --mm-per-px   Fallback if you already know the scale.
  neither       Prints pixels only.
"""
import argparse, csv, statistics as st
import xml.etree.ElementTree as ET

GOOD_MM, WATCH_MM = 5.0, 6.0


def parse(xml_path):
    root = ET.parse(xml_path).getroot()
    rows = []
    for img in root.iter("image"):
        name = img.get("name")
        for box in img.iter("box"):
            xtl, ytl = float(box.get("xtl")), float(box.get("ytl"))
            xbr, ybr = float(box.get("xbr")), float(box.get("ybr"))
            w, h = xbr - xtl, ybr - ytl
            rows.append({
                "image": name,
                "label": box.get("label"),
                "cx": round((xtl + xbr) / 2, 1),
                "cy": round((ytl + ybr) / 2, 1),
                "width_px": round(min(w, h), 2),
                "length_px": round(max(w, h), 2),
            })
    return rows


def measure_pitch(rows):
    """Median centre-to-centre spacing, per image, along the short axis."""
    pitches = []
    by_img = {}
    for r in rows:
        by_img.setdefault(r["image"], []).append(r)

    for name, rs in by_img.items():
        if len(rs) < 4:
            continue
        # holes are elongated: spacing runs across the short dimension
        xs = sorted(r["cx"] for r in rs)
        d = [b - a for a, b in zip(xs, xs[1:]) if b - a > 1]
        if len(d) < 3:
            continue
        m = st.median(d)
        d = [v for v in d if 0.7 * m < v < 1.3 * m]   # drop gaps between plates
        if d:
            pitches.append(st.median(d))
    return st.median(pitches) if pitches else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml")
    ap.add_argument("--pitch-mm", type=float)
    ap.add_argument("--mm-per-px", type=float)
    ap.add_argument("--out", default="holes.csv")
    a = ap.parse_args()

    rows = parse(a.xml)
    if not rows:
        raise SystemExit("no boxes found - did you export 'CVAT for images 1.1'?")

    widths = [r["width_px"] for r in rows]
    pitch_px = measure_pitch(rows)

    print(f"holes            : {len(rows)}")
    print(f"images           : {len(set(r['image'] for r in rows))}")
    print(f"width px  median : {st.median(widths):.2f}")
    print(f"width px  min/max: {min(widths):.2f} / {max(widths):.2f}")
    if pitch_px:
        print(f"pitch px  median : {pitch_px:.2f}")

    mmpp = None
    if a.pitch_mm:
        if not pitch_px:
            raise SystemExit("could not measure pitch - annotate more holes per image")
        mmpp = a.pitch_mm / pitch_px
        print(f"scale            : {mmpp:.5f} mm/px  (from pitch)")
    elif a.mm_per_px:
        mmpp = a.mm_per_px
        print(f"scale            : {mmpp:.5f} mm/px  (given)")
    else:
        print("scale            : none - pixels only. pass --pitch-mm")

    if mmpp:
        for r in rows:
            r["mm"] = round(r["width_px"] * mmpp, 2)
            r["grade"] = ("GOOD" if r["mm"] < GOOD_MM else
                          "WATCH" if r["mm"] <= WATCH_MM else "REJECT")
        mms = [r["mm"] for r in rows]
        rej = 100 * sum(r["grade"] == "REJECT" for r in rows) / len(rows)
        print()
        print(f"width mm  median : {st.median(mms):.2f}")
        print(f"width mm  p95    : {sorted(mms)[int(.95 * len(mms)) - 1]:.2f}")
        print(f"reject %         : {rej:.1f}")
        print(f"VERDICT          : {'REJECT' if rej > 10 else 'WATCH' if st.median(mms) > GOOD_MM else 'PASS'}")

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
