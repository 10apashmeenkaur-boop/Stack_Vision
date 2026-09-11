"""
api.py - the full backend. Everything the frontend needs, over HTTP.

    pip install fastapi uvicorn python-multipart
    python api.py
    -> http://localhost:8090/docs

No CLI steps required. The frontend can upload a video, pull a frame,
let the user click zones in the browser, start a run, poll it, and pull
results - all over HTTP.

Endpoints
  GET  /api/health                      is it alive
  GET  /api/config                      thresholds and defaults
  POST /api/videos                      upload a video          -> video_id
  GET  /api/videos                      list uploaded videos
  GET  /api/videos/{id}                 metadata (size, fps, frames)
  GET  /api/videos/{id}/frame?n=100     a JPEG frame, for annotating
  POST /api/videos/{id}/zones           save zones (clicked in browser)
  GET  /api/videos/{id}/zones           read them back
  POST /api/videos/{id}/autozones       detect zones automatically
  POST /api/runs                        start an inspection     -> run_id
  GET  /api/runs                        list runs
  GET  /api/runs/{id}                   status + summary (poll this)
  GET  /api/runs/{id}/plates            all plate results
  GET  /api/runs/{id}/frame             latest annotated JPEG
  GET  /api/runs/{id}/report.csv        download
  DELETE /api/runs/{id}                 stop / remove
  WS   /api/runs/{id}/live              push updates (optional)
"""
import io, csv, json, os, time, uuid, threading
import cv2
import numpy as np
import statistics as st
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket
from fastapi.responses import Response, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

import gauge
import openarea
from video_gauge import vprofile, profile_shift

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)

app = FastAPI(title="Hira Vision API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

VIDEOS = {}     # video_id -> {path, name, zones, meta}
RUNS = {}       # run_id   -> {status, progress, plates, latest_jpeg, ...}


# ----------------------------------------------------------------- models
class Zones(BaseModel):
    plates: List[List[List[float]]]           # [[[x,y] x4], ...]
    plate_mm: List[float] = [315.0, 255.0]


class RunRequest(BaseModel):
    video_id: str
    dark: float = 0.35
    slots: int = 39
    nominal_mm: float = 4.0
    reject_mm: float = 6.0
    every: int = 4
    min_obs: int = 8
    px_per_mm: float = 2.0
    len_range: List[float] = [55.0, 90.0]


# ----------------------------------------------------------------- basics
@app.get("/")
async def root():
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("app.html", "index.html"):
        p = os.path.join(here, name)
        if os.path.exists(p):
            return HTMLResponse(open(p, encoding="utf-8").read())
    return HTMLResponse("<h2>Hira Vision API</h2>"
                        "<p>See <a href='/docs'>/docs</a></p>")


@app.get("/api/health")
async def health():
    return {"ok": True, "videos": len(VIDEOS), "runs": len(RUNS),
            "time": time.time()}


@app.get("/api/config")
async def config():
    return {
        "plate_mm": [gauge.PLATE_W_MM, gauge.PLATE_H_MM],
        "thresholds": {"pass_max_mm": 5.0, "watch_max_mm": 6.0},
        "defaults": RunRequest(video_id="").model_dump(),
        "notes": {
            "verdict": "PASS <=5mm, WATCH <=6mm, REJECT >6mm",
            "stdev_mm": "quality meter, under 0.35 is good",
            "slot_len_mm": "should be 62-80mm; outside that the zone is wrong",
        }}


# ----------------------------------------------------------------- videos
@app.post("/api/videos")
async def upload_video(file: UploadFile = File(...)):
    vid = uuid.uuid4().hex[:12]
    path = os.path.join(DATA, f"{vid}_{file.filename}")
    with open(path, "wb") as f:
        f.write(await file.read())
    cap = cv2.VideoCapture(path)
    meta = {"fps": round(cap.get(cv2.CAP_PROP_FPS), 2),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(3)), "height": int(cap.get(4))}
    cap.release()
    if meta["frames"] <= 0:
        os.remove(path)
        raise HTTPException(400, "cannot decode this video (try H.264 mp4)")
    VIDEOS[vid] = {"path": path, "name": file.filename,
                   "meta": meta, "zones": None}
    return {"video_id": vid, "name": file.filename, **meta}


@app.get("/api/videos")
async def list_videos():
    return [{"video_id": k, "name": v["name"], **v["meta"],
             "has_zones": v["zones"] is not None} for k, v in VIDEOS.items()]


@app.get("/api/videos/{vid}")
async def get_video(vid: str):
    v = VIDEOS.get(vid)
    if not v:
        raise HTTPException(404, "unknown video_id")
    return {"video_id": vid, "name": v["name"], **v["meta"],
            "zones": v["zones"]}


@app.get("/api/videos/{vid}/frame")
async def get_frame(vid: str, n: int = 100, scale: float = 1.0):
    """A JPEG frame. The frontend shows this and the user clicks corners."""
    v = VIDEOS.get(vid)
    if not v:
        raise HTTPException(404, "unknown video_id")
    cap = cv2.VideoCapture(v["path"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n - 1))
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(400, f"cannot read frame {n}")
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return Response(buf.tobytes(), media_type="image/jpeg")


@app.post("/api/videos/{vid}/zones")
async def set_zones(vid: str, z: Zones):
    """Save zones clicked in the browser. 4 corners each: TL, TR, BR, BL."""
    v = VIDEOS.get(vid)
    if not v:
        raise HTTPException(404, "unknown video_id")
    for q in z.plates:
        if len(q) != 4 or any(len(p) != 2 for p in q):
            raise HTTPException(422, "each zone needs exactly 4 [x,y] points")
    v["zones"] = {"plates": z.plates, "plate_mm": z.plate_mm}
    return {"video_id": vid, "zones": len(z.plates)}


@app.get("/api/videos/{vid}/zones")
async def get_zones(vid: str):
    v = VIDEOS.get(vid)
    if not v:
        raise HTTPException(404, "unknown video_id")
    if not v["zones"]:
        raise HTTPException(404, "no zones saved for this video")
    return v["zones"]


@app.post("/api/videos/{vid}/autozones")
async def auto_zones(vid: str, frame: int = 100, px_per_mm: float = 2.0):
    """Detect zones automatically. May fail on some cameras - then click."""
    v = VIDEOS.get(vid)
    if not v:
        raise HTTPException(404, "unknown video_id")
    try:
        import autozone
    except ImportError:
        raise HTTPException(501, "autozone.py not available")
    gauge.set_plate(315.0, 255.0, px_per_mm)
    cap = cv2.VideoCapture(v["path"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(400, "cannot read frame")
    H, W = img.shape[:2]
    roi = (int(.12*W), int(.04*H), int(.82*W), int(.78*H))
    vx = autozone.vertical_seams(img, roi)
    if len(vx) < 2:
        raise HTTPException(422, "no seams found - annotate manually")
    found = []
    coarse = max(14, H // 40)
    for i in range(len(vx) - 1):
        xa, xb = vx[i], vx[i+1]
        if xb - xa < 80:
            continue
        best = (-1, None, None)
        for h in range(int(.10*H), int(.34*H), coarse):
            for y in range(roi[1], roi[3] - h, coarse):
                q = [[xa, y], [xb, y], [xb, y+h], [xa, y+h]]
                s, info = autozone.score(img, q, 0.35, 39)
                if s > best[0]:
                    best = (s, q, info)
        if best[0] >= 0.55:
            found.append({"quad": [[float(x), float(y)] for x, y in best[1]],
                          "score": round(best[0], 3), "info": best[2]})
    if not found:
        raise HTTPException(422, "nothing scored well - annotate manually")
    VIDEOS[vid]["zones"] = {"plates": [f["quad"] for f in found],
                            "plate_mm": [315.0, 255.0]}
    return {"video_id": vid, "zones": found}


# ------------------------------------------------------------------- runs
def _worker(run_id: str, req: RunRequest):
    R = RUNS[run_id]
    v = VIDEOS[req.video_id]
    z = v["zones"]
    gauge.set_plate(*z.get("plate_mm", [315.0, 255.0]), req.px_per_mm)
    PPM = gauge.PX_PER_MM
    zones = z["plates"]
    Hs = [gauge.homography_from_corners(q) for q in zones]
    heights = [max(p[1] for p in q) - min(p[1] for p in q) for q in zones]

    cap = cv2.VideoCapture(v["path"])
    W, Hh = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    buf = [[] for _ in zones]; pid = [0]*len(zones); travel = [0.0]*len(zones)
    prevp = None; idx = 0
    R.update(status="running", total=total)

    while True:
        if R.get("cancel"):
            R["status"] = "cancelled"; break
        ok, frame = cap.read()
        if not ok:
            R["status"] = "done"; break
        idx += 1
        if (idx - 1) % req.every:
            continue
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        band = g[int(.15*Hh):int(.60*Hh), int(.30*W):int(.70*W)]
        cp = vprofile(band)
        dy = 0.0 if prevp is None else abs(profile_shift(prevp, cp))
        prevp = cp

        vis = frame.copy(); live = []
        for zi, (q, Hm) in enumerate(zip(zones, Hs)):
            rect = cv2.warpPerspective(frame, Hm, (gauge.CW, gauge.CH))
            area, keep, _, len_px = openarea.open_area(rect, dark_frac=req.dark)
            slen = len_px / PPM if len_px > 0 else 0
            if slen and req.len_range[0] <= slen <= req.len_range[1]:
                buf[zi].append((area/(PPM*PPM))/(req.slots*slen))
            travel[zi] += dy
            if travel[zi] >= heights[zi] and len(buf[zi]) >= req.min_obs:
                med = st.median(buf[zi]); pid[zi] += 1
                R["plates"].append({
                    "zone": zi+1, "plate": pid[zi], "frame": idx,
                    "n_obs": len(buf[zi]), "width_mm": round(med, 2),
                    "stdev_mm": round(st.pstdev(buf[zi]), 3)
                    if len(buf[zi]) > 1 else 0.0,
                    "slot_len_mm": round(slen, 1),
                    "verdict": ("REJECT" if med > req.reject_mm else
                                "WATCH" if med > req.nominal_mm*1.25 else "PASS")})
                buf[zi] = []; travel[zi] = 0.0
            cur = st.median(buf[zi]) if buf[zi] else None
            col = ((120,120,120) if cur is None else
                   (0,200,0) if cur <= req.nominal_mm*1.25 else
                   (0,210,255) if cur <= req.reject_mm else (0,0,255))
            cv2.polylines(vis, [np.array(q, np.int32)], True, col, 3)
            tag = f"Z{zi+1} " + (f"{cur:.2f}mm" if cur else "...")
            org = (int(q[0][0])+6, int(q[0][1])+28)
            cv2.putText(vis, tag, org, 0, .8, (0,0,0), 5, cv2.LINE_AA)
            cv2.putText(vis, tag, org, 0, .8, col, 2, cv2.LINE_AA)
            live.append({"zone": zi+1, "mm": round(cur,2) if cur else None,
                         "n": len(buf[zi]), "slot_len_mm": round(slen,1)})

        _, jb = cv2.imencode(".jpg", cv2.resize(vis, (W//3, Hh//3)),
                             [cv2.IMWRITE_JPEG_QUALITY, 70])
        R["latest_jpeg"] = jb.tobytes()
        R.update(frame=idx, belt_px=round(dy,2), live=live)
    cap.release()
    R["finished_at"] = time.time()


@app.post("/api/runs")
async def start_run(req: RunRequest):
    v = VIDEOS.get(req.video_id)
    if not v:
        raise HTTPException(404, "unknown video_id")
    if not v["zones"]:
        raise HTTPException(409, "no zones for this video - POST zones first, "
                                 "or call /autozones")
    rid = uuid.uuid4().hex[:12]
    RUNS[rid] = {"run_id": rid, "video_id": req.video_id, "status": "starting",
                 "frame": 0, "total": v["meta"]["frames"], "belt_px": 0.0,
                 "live": [], "plates": [], "latest_jpeg": None,
                 "started_at": time.time(), "params": req.model_dump()}
    threading.Thread(target=_worker, args=(rid, req), daemon=True).start()
    return {"run_id": rid, "status": "starting"}


@app.get("/api/runs")
async def list_runs():
    return [{k: r[k] for k in ("run_id","video_id","status","frame","total")}
            | {"plates": len(r["plates"])} for r in RUNS.values()]


@app.get("/api/runs/{rid}")
async def run_status(rid: str):
    r = RUNS.get(rid)
    if not r:
        raise HTTPException(404, "unknown run_id")
    ws = [p["width_mm"] for p in r["plates"]]
    return {"run_id": rid, "status": r["status"],
            "frame": r["frame"], "total": r["total"],
            "progress": round(100*r["frame"]/max(1,r["total"]), 1),
            "belt_px": r["belt_px"], "live": r["live"],
            "summary": {"plates": len(ws),
                        "median_mm": round(st.median(ws),2) if ws else None,
                        "reject_pct": round(100*sum(
                            p["verdict"]=="REJECT" for p in r["plates"])/len(ws),1)
                        if ws else 0.0},
            "recent": r["plates"][-10:]}


@app.get("/api/runs/{rid}/plates")
async def run_plates(rid: str):
    r = RUNS.get(rid)
    if not r:
        raise HTTPException(404, "unknown run_id")
    return {"run_id": rid, "count": len(r["plates"]), "plates": r["plates"]}


@app.get("/api/runs/{rid}/frame")
async def run_frame(rid: str):
    r = RUNS.get(rid)
    if not r:
        raise HTTPException(404, "unknown run_id")
    if not r["latest_jpeg"]:
        raise HTTPException(404, "no frame yet")
    return Response(r["latest_jpeg"], media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.get("/api/runs/{rid}/report.csv")
async def run_csv(rid: str):
    r = RUNS.get(rid)
    if not r:
        raise HTTPException(404, "unknown run_id")
    if not r["plates"]:
        return Response("no plates measured yet", media_type="text/plain")
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=list(r["plates"][0].keys()))
    w.writeheader(); w.writerows(r["plates"])
    return Response(out.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition":
                             f"attachment; filename=hira_{rid}.csv"})


@app.delete("/api/runs/{rid}")
async def stop_run(rid: str):
    r = RUNS.get(rid)
    if not r:
        raise HTTPException(404, "unknown run_id")
    r["cancel"] = True
    return {"run_id": rid, "status": "cancelling"}


@app.websocket("/api/runs/{rid}/live")
async def run_live(sock: WebSocket, rid: str):
    import asyncio, base64
    await sock.accept()
    r = RUNS.get(rid)
    if not r:
        await sock.close(); return
    last = -1
    try:
        while r["status"] in ("starting", "running"):
            if r["frame"] != last:
                last = r["frame"]
                await sock.send_json({
                    "type": "frame", "frame": r["frame"], "total": r["total"],
                    "belt_px": r["belt_px"], "live": r["live"],
                    "recent": r["plates"][-10:],
                    "annotated": base64.b64encode(r["latest_jpeg"]).decode()
                    if r["latest_jpeg"] else None})
            await asyncio.sleep(0.05)
        await sock.send_json({"type": "done", "plates": len(r["plates"])})
    except Exception:
        pass


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--preload", nargs="*", default=None,
                    help="video files to register at startup")
    ap.add_argument("--zones", default=None,
                    help="zones.json to attach to the first preloaded video")
    a = ap.parse_args()

    for p in (a.preload or []):
        if not os.path.exists(p):
            print(f"  skip {p} (not found)"); continue
        vid = uuid.uuid4().hex[:12]
        cap = cv2.VideoCapture(p)
        meta = {"fps": round(cap.get(cv2.CAP_PROP_FPS), 2),
                "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "width": int(cap.get(3)), "height": int(cap.get(4))}
        cap.release()
        zones = None
        if a.zones and os.path.exists(a.zones):
            z = json.load(open(a.zones))
            zones = {"plates": z["plates"],
                     "plate_mm": z.get("plate_mm", [315.0, 255.0])}
        VIDEOS[vid] = {"path": os.path.abspath(p), "name": os.path.basename(p),
                       "meta": meta, "zones": zones}
        print(f"  preloaded {p}  video_id={vid}  zones="
              f"{len(zones['plates']) if zones else 0}")

    print(f"\nHira Vision API  ->  http://localhost:{a.port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=a.port, log_level="warning")
