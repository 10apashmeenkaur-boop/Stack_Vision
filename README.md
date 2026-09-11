# Hira Vision — grate slot inspection

Measures slot width in millimetres on rotating grate plates from CCTV, and
flags plates past tolerance.

## Install

```powershell
pip install opencv-python numpy fastapi uvicorn
```

## Quick start

```powershell
# 1. find the measurement zones (once per camera, ~15 s)
python autozone.py clip.mp4 --px-per-mm 2

# 2. check autozones.png — three green boxes should sit on real plates

# 3a. run over a video, CSV out
python video_gauge.py clip.mp4 --zones zones.json --px-per-mm 2 --every 4 --no-window

# 3b. or run the web UI
python server.py clip.mp4 --zones zones.json
#     then open http://localhost:8000
```

Zones are tied to the **camera position**, not the file. Same camera, any
clip — reuse `zones.json` forever. Only a moved camera needs a rerun.

## How it works

```
frame
 └─ homography per zone ──► flat canvas, fixed px/mm
      └─ global threshold ─► open-area mask (the slots)
           └─ mean width = open_area / (slots × slot_length)
                └─ buffer across frames
                     └─ belt moved one plate height? flush as ONE plate result
```

**Why open area, not per-hole detection.** Detecting individual slots found
only 20–32 of 39, and *which* it missed varied by plate, so each plate's
average came from a different biased subset. Open area counts dark pixels: a
slot split by debris, merged with a neighbour or half-glared still contributes
its true area. Recall stops mattering.

**Why slot length is the validation.** Scale is set from the plate's 315 mm
width. Slot length is a separate dimension from the drawing (62 / 79 / 62 mm
row bands) that the calibration never touched — so if measured lengths land
near those, the millimetres are real. Every tool prints `slot_len_mm`. It is
the first number to read.

## Scripts

| Script | Purpose |
|---|---|
| `autozone.py` | Finds zones automatically. No clicking. |
| `annotate_plates.py` | Manual zone clicking, if auto misses. |
| `video_gauge.py` | Runs a whole video → one row per plate. |
| `server.py` | Web UI, live. |
| `openarea.py` | Single-frame measurement (also the core library). |
| `gauge.py` | Homography, rectification, per-hole measurement. |
| `run_gauge.py` | Single-plate diagnostics with full stage output. |

## Key flags

| Flag | Meaning |
|---|---|
| `--px-per-mm 2` | Canvas resolution. 2 is ~4× faster than 4 and costs little for area measurement. Use it. |
| `--every 4` | Process every 4th frame. |
| `--dark 0.35` | Threshold between darkest pixel and plate face. Lower = only very dark counts as open. |
| `--len-range 60 80` | Drop frames whose slot length is implausible — bad geometry rejected instead of averaged in. |
| `--min-obs 20` | Frames required before a plate result is emitted. |
| `--slots 39` | Slots per plate, from the drawing. |
| `--nominal-mm 4.0` | New-plate slot width. |
| `--reject-mm 6.0` | Reject threshold. |

## Reading the output

`plates_timeline.csv` — one row per physical plate:

| column | meaning |
|---|---|
| `width_mm` | mean slot width, median over all frames that plate was visible |
| `n_obs` | how many frames backed it |
| `stdev_mm` | **quality meter.** Under 0.35 mm is good. |
| `verdict` | PASS / WATCH / REJECT |

Check in this order: `slot_len_mm` (62–80?), then `stdev_mm` (<0.35?), then
`width_mm`. If the first two are wrong the third is meaningless.

## Validation achieved

| Check | Result |
|---|---|
| Slot length vs drawing | 62.5 / 62.5 / 63.0 mm vs 62 mm |
| Mean width vs 4 mm nominal, healthy plates | 4.55–4.66 mm |
| Plate-to-plate spread | 0.81–0.86 mm |
| Per-plate stdev across frames | 0.29–0.34 mm |
| Belt speed cross-check | 1.97 m/min, within the stated 1–3 m/min |

## Known limits

- **Fisheye not corrected.** Needs a chessboard set from the plant camera
  (`cv2.calibrateCamera` → `cv2.undistort` before the homography). Would
  tighten the plate-to-plate spread.
- **Absolute zero not set.** Width is the *dark opening*, which includes the
  slot chamfer, so it reads wider than a caliper on the through-slot. One
  caliper measurement on a healthy plate gives the offset.
- **±0.86 mm plate to plate** means the 5–6 mm band is not resolvable. A
  healthy plate is clearly distinguishable from a rejected one; 5.5 vs 6.0 is
  not.
- **Horizontal seams are not detectable** (curved by the lens). `autozone.py`
  searches the vertical extent instead of detecting it.

## Troubleshooting

**MemoryError** — add `--px-per-mm 2`.

**Slot length way off 62 mm** — the zone spans two plates or sits on a seam.
Rerun `autozone.py`, or click by hand with `annotate_plates.py`.

**"no plates completed"** — the belt didn't travel a full plate height in that
clip. Lower `--min-obs`.

**PermissionError on a CSV** — it's open in Excel. Close it.

**Slow** — `--px-per-mm 2 --every 4 --no-window`.
