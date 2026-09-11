# Hira Vision — backend handover

Computer-vision gauge that measures grate slot width in millimetres from
plant CCTV and flags worn plates.

This document is for whoever is building the frontend. The Python backend is
done and working; you need the API contract, which is at the bottom.

---

## 1. Run it in five minutes

```bash
pip install opencv-python numpy fastapi uvicorn

# one-time per camera: click the 3 measurement zones
python annotate_plates.py clip.mp4 --out zones.json

# start the API
python server.py clip.mp4 --zones zones.json
# -> http://localhost:8000
```

`index.html` is a working reference frontend. Open the page and you will see
exactly what the API emits. Replace it with your own app; the contract below
is all you need.

For a live camera, pass an RTSP URL instead of a filename. Nothing else
changes.

```bash
python server.py rtsp://user:pass@10.0.0.5/stream --zones zones.json
```

---

## 2. What the system does

```
video frame
   │
   ├─ for each ZONE (a fixed quad in the image):
   │     homography  ──► flat canvas at a fixed px/mm
   │     threshold   ──► mask of the open slots
   │     mean width   =  open_area / (slot_count × slot_length)
   │
   ├─ belt travel tracked by profile correlation
   │
   └─ when the belt has moved one plate height, the plate in that zone has
      been replaced → flush the buffered readings as ONE plate result
      (the median of every frame that plate was visible for)
```

**Zones, not plates.** The camera is fixed and the plates scroll past it. So
you annotate fixed regions once, and every plate that passes through a region
gets measured there. `zones.json` is valid for that camera forever — only a
moved camera needs re-annotating.

**Why open area instead of detecting each slot.** Per-slot detection found
only 20–32 of 39 slots, and *which* ones it missed varied by plate, so every
plate's average came from a different biased subset. Counting open pixels has
no such failure: a slot split by debris, merged with a neighbour, or partly
glared still contributes its true area.

---

## 3. Accuracy, honestly

| Check | Result |
|---|---|
| Slot length vs engineering drawing | 62.5 / 62.5 / 63.0 mm vs 62 mm spec |
| Mean width on known-healthy plates | 4.55–4.66 mm vs 4.0 mm nominal |
| Plate-to-plate spread | 0.81–0.86 mm |
| Repeatability across frames (per plate) | 0.29–0.34 mm |
| Belt speed cross-check | 1.97 m/min, within the stated 1–3 m/min |

**Self-validating.** Scale is derived from the plate's 315 mm width. Slot
*length* is an independent dimension from the drawing that the calibration
never touches — so if measured lengths land near 62 mm, the millimetres are
real. Every response carries this; surface it in the UI.

**Known limits — do not hide these in the product:**

- Fisheye distortion is not corrected (needs a chessboard calibration set)
- Absolute zero is not set: width is the *dark opening*, which includes the
  slot chamfer, so it reads wider than a caliper. One caliper measurement on
  a healthy plate gives the offset, applied via `--nominal-mm`
- ±0.86 mm plate-to-plate means the 5–6 mm band is not resolvable. Healthy
  vs rejected is clear; 5.5 vs 6.0 is not
- **Requires roughly 10 px across a slot.** The drum CCTV gives 0.25 mm/px
  and works. A 476×848 conveyor clip gives 2.25 mm/px — a 4 mm slot is under
  2 pixels there, and no software recovers that

---

## 4. API

Base URL `http://localhost:8000`.

### GET `/`
Serves `index.html`. Replace with your own app.

### GET `/report.csv`
All plate results so far, as CSV. `Content-Disposition: attachment`.
Wire straight to a download button.

### WS `/ws`
Opens the run. Server pushes one message per processed frame. The client
sends nothing.

**Frame message**

```json
{
  "type": "frame",
  "frame": 412,
  "total": 943,
  "belt_px": 1.45,
  "original":  "<base64 jpeg>",
  "annotated": "<base64 jpeg>",
  "live": [
    { "zone": 1, "mm": 4.61, "n": 23 },
    { "zone": 2, "mm": 4.48, "n": 19 },
    { "zone": 3, "mm": null, "n": 0 }
  ],
  "results": [
    { "zone": 1, "plate": 7, "frame": 388, "n_obs": 44,
      "width_mm": 4.87, "stdev_mm": 0.34, "verdict": "PASS" }
  ],
  "summary": { "plates": 14, "median_mm": 4.97, "reject_pct": 7.1 }
}
```

**Completion message**

```json
{ "type": "done", "plates": 14 }
```

### Field reference

| Field | Type | Meaning |
|---|---|---|
| `frame` / `total` | int | progress. `total` is −1 for a live stream |
| `belt_px` | float | belt travel this frame, pixels |
| `original` | b64 jpeg | raw frame, downscaled |
| `annotated` | b64 jpeg | zones drawn, colour-coded, mm labelled |
| `live[].zone` | int | 1-based zone index |
| `live[].mm` | float\|null | running median for the plate currently in that zone. **null until `n` reaches min_obs** |
| `live[].n` | int | frames buffered for the current plate |
| `results[]` | array | completed plates, most recent 12 |
| `results[].plate` | int | per-zone counter, increments as plates pass |
| `results[].n_obs` | int | frames the result is the median of |
| `results[].stdev_mm` | float | **quality meter. Under 0.35 is good** |
| `results[].verdict` | enum | `PASS` \| `WATCH` \| `REJECT` |
| `summary.reject_pct` | float | share of completed plates rejected |

### Verdict thresholds

```
width_mm <= 5.0   PASS      (nominal 4.0 mm)
width_mm <= 6.0   WATCH
width_mm  > 6.0   REJECT
```

Set server-side via `--nominal-mm` and `--reject-mm`. Don't hardcode in the
frontend — read `verdict`.

---

## 5. Frontend notes

**`live[].mm` is null at the start.** A zone has no reading until enough
frames buffer. Render a dash, not a zero.

**`results` grows slowly.** A plate completes when the belt has travelled one
plate height — roughly every 150–250 frames. Don't expect a row per frame.
Only the last 12 are sent; keep your own array if you want full history.

**Images are base64 JPEG.** `<img src={"data:image/jpeg;base64," + msg.original}>`.
At ~15 fps this is fine over localhost. For remote, ask and I'll switch to an
MJPEG endpoint.

**The dashboard should show `stdev_mm` and slot length.** They're what make
the numbers defensible rather than asserted. A UI that shows only a verdict
is less trustworthy than one that shows its own confidence.

### Suggested layout

```
┌──────────────┬──────────────┐
│   Source     │   Measured   │   two video panels
├──────────────┴──────────────┤
│  plates · median · reject%  │   summary tiles
├─────────────────────────────┤
│  Z1 4.61   Z2 4.48   Z3 —   │   live per-zone
├─────────────────────────────┤
│  zone plate n width sd ✓/✗  │   results table
└─────────────────────────────┘
```

---

## 6. If you prefer REST over WebSocket

Say so and these get added:

```
POST /api/run        {video, zones}  → {run_id}
GET  /api/status/:id                 → progress + summary
GET  /api/plates/:id                 → all results
GET  /api/frame/:id                  → latest annotated jpeg
```

WebSocket is smoother for the live view since it's push, no polling lag.

---

## 7. Files

| File | Role |
|---|---|
| `server.py` | FastAPI backend — the API above |
| `index.html` | reference frontend |
| `gauge.py` | homography, rectification, per-hole measurement |
| `openarea.py` | the measurement core |
| `video_gauge.py` | CLI: whole video → one row per plate |
| `annotate_plates.py` | click zones once per camera |
| `autozone.py` | automatic zone selection (experimental) |
| `zones.json` | saved zones — reusable for that camera forever |

**Do not edit `gauge.py` or `openarea.py`.** The constants in them were fitted
against the engineering drawing and validated. Changing them silently
invalidates the accuracy figures above.

---

## 8. Not done yet

- Lens undistortion (chessboard calibration required)
- Caliper reference to set absolute zero
- Multi-camera support in one session
- Persistence — results are in memory and reset when the server restarts
- Auth
