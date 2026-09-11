"""
server.py - web backend. Streams the gauge live to a browser.

    pip install fastapi uvicorn
    python server.py clip.mp4 --zones zones.json
    open http://localhost:8000

For a live camera, pass the RTSP URL instead of a file:
    python server.py rtsp://user:pass@10.0.0.5/stream --zones zones.json
Nothing else changes - VideoCapture handles both.
"""
import argparse, asyncio, base64, io, json, csv
import cv2
import numpy as np
import statistics as st
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
import uvicorn

import gauge
import openarea
from video_gauge import vprofile, profile_shift

app = FastAPI()
CFG = {}
RESULTS = []


def jpg(img, q=70):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf).decode() if ok else ""


@app.get("/")
async def index():
    import os
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    if not os.path.exists(p):
        return HTMLResponse(
            "<h2>index.html not found</h2>"
            f"<p>Expected it at:<br><code>{p}</code></p>"
            "<p>Put index.html in the same folder as server.py.</p>"
            "<p>The API still works: "
            "<a href='/report.csv'>/report.csv</a> and ws://localhost:8000/ws"
            "</p>", status_code=200)
    return HTMLResponse(open(p, encoding="utf-8").read())


@app.get("/report.csv")
async def report():
    if not RESULTS:
        return Response("no data yet", media_type="text/plain")
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=list(RESULTS[0].keys()))
    w.writeheader(); w.writerows(RESULTS)
    return Response(out.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition":
                             "attachment; filename=hira_report.csv"})


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    a = CFG["args"]
    zones = CFG["zones"]
    Hs = [gauge.homography_from_corners(z) for z in zones]
    heights = [max(p[1] for p in z) - min(p[1] for p in z) for z in zones]
    PPM = gauge.PX_PER_MM

    cap = cv2.VideoCapture(a.video)
    W = int(cap.get(3)); Hh = int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    buf = [[] for _ in zones]
    pid = [0] * len(zones)
    travel = [0.0] * len(zones)
    prevp = None
    idx = 0
    RESULTS.clear()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                await sock.send_json({"type": "done",
                                      "plates": len(RESULTS)})
                break
            idx += 1
            if (idx - 1) % a.every:
                continue

            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            band = g[int(.15*Hh):int(.60*Hh), int(.30*W):int(.70*W)]
            cp = vprofile(band)
            dy = 0.0 if prevp is None else abs(profile_shift(prevp, cp))
            prevp = cp

            vis = frame.copy()
            live = []
            for zi, (z, Hm) in enumerate(zip(zones, Hs)):
                rect = cv2.warpPerspective(frame, Hm, (gauge.CW, gauge.CH))
                area, keep, _, len_px = openarea.open_area(rect,
                                                           dark_frac=a.dark)
                if len_px > 0:
                    L = len_px / PPM
                    if a.len_range[0] <= L <= a.len_range[1]:
                        buf[zi].append((area / (PPM*PPM)) / (a.slots * L))

                travel[zi] += dy
                if travel[zi] >= heights[zi] and len(buf[zi]) >= a.min_obs:
                    med = st.median(buf[zi])
                    pid[zi] += 1
                    RESULTS.append({
                        "zone": zi + 1, "plate": pid[zi], "frame": idx,
                        "n_obs": len(buf[zi]), "width_mm": round(med, 2),
                        "stdev_mm": round(st.pstdev(buf[zi]), 3)
                        if len(buf[zi]) > 1 else 0,
                        "verdict": ("REJECT" if med > a.reject_mm else
                                    "WATCH" if med > a.nominal_mm * 1.25
                                    else "PASS")})
                    buf[zi] = []; travel[zi] = 0.0

                cur = st.median(buf[zi]) if buf[zi] else None
                col = ((120, 120, 120) if cur is None else
                       (0, 200, 0) if cur <= a.nominal_mm * 1.25 else
                       (0, 210, 255) if cur <= a.reject_mm else (0, 0, 255))
                cv2.polylines(vis, [np.array(z, np.int32)], True, col, 3)
                tag = f"Z{zi+1} " + (f"{cur:.2f}mm" if cur else "...")
                org = (int(z[0][0]) + 6, int(z[0][1]) + 28)
                cv2.putText(vis, tag, org, 0, .8, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(vis, tag, org, 0, .8, col, 2, cv2.LINE_AA)
                live.append({"zone": zi + 1,
                             "mm": round(cur, 2) if cur else None,
                             "n": len(buf[zi])})

            sm = cv2.resize(frame, (W // 3, Hh // 3))
            an = cv2.resize(vis, (W // 3, Hh // 3))
            ws_ = [r["width_mm"] for r in RESULTS]
            await sock.send_json({
                "type": "frame", "frame": idx, "total": total,
                "belt_px": round(dy, 2), "live": live,
                "original": jpg(sm), "annotated": jpg(an),
                "results": RESULTS[-12:],
                "summary": {
                    "plates": len(RESULTS),
                    "median_mm": round(st.median(ws_), 2) if ws_ else None,
                    "reject_pct": round(
                        100 * sum(r["verdict"] == "REJECT" for r in RESULTS)
                        / len(RESULTS), 1) if ws_ else 0,
                }})
            await asyncio.sleep(0.001)
    except WebSocketDisconnect:
        pass
    finally:
        cap.release()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--zones", default="zones.json")
    ap.add_argument("--dark", type=float, default=0.35)
    ap.add_argument("--slots", type=int, default=39)
    ap.add_argument("--nominal-mm", type=float, default=4.0)
    ap.add_argument("--reject-mm", type=float, default=6.0)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--min-obs", type=int, default=8)
    ap.add_argument("--len-range", type=float, nargs=2, default=[55.0, 90.0])
    ap.add_argument("--px-per-mm", type=float, default=2.0)
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()

    cfg = json.load(open(a.zones))
    gauge.set_plate(*cfg.get("plate_mm", [315.0, 255.0]), a.px_per_mm)
    CFG["args"] = a
    CFG["zones"] = cfg["plates"]
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(here, "index.html")):
        print("WARNING: index.html is not next to server.py - the page will "
              "not load. Put it in " + here)
    print(f"zones {len(cfg['plates'])}  ->  http://localhost:{a.port}")
    uvicorn.run(app, host="0.0.0.0", port=a.port, log_level="warning")
