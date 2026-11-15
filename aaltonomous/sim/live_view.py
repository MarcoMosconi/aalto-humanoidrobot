"""Live demo: stream the Isaac Sim GUI to a browser and run eval episodes of a submission image from the page.

    # 1. X server on the GPU (no monitor needed), then the sim with its GUI on it
    Xorg :1 -config aaltonomous/sim/xorg_headless_nvidia.conf -noreset -nolisten tcp &
    DISPLAY=:1 TASK=Aalto-Sort-G129-Dex3-Joint bash aaltonomous/sim/run_sim.sh &
    # 2. this server (keeps the image running, so episodes start at once)
    python -m aaltonomous.sim.live_view --container ghcr.io/marcomosconi/upsala-v5:v1 --team upsala
    # 3. open http://localhost:8770   (remote: ssh -L 8770:localhost:8770 <host>)

The video is the X display itself (ffmpeg x11grab -> MJPEG), i.e. exactly what Isaac Sim renders, unedited.
"Run episode" starts `aaltonomous.sim.evaluate --policy ws://localhost:<port> --trials 1 --seed N` (the stock
evaluator, same scoring and gate as the sim gate) against the running image; results appear in the side panel.
--view viewport crops the stream to the 3D viewport (window maximised to 1920x1080 at start).
"""
from __future__ import annotations

import argparse
import ctypes as C
import json
import logging
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO = Path(__file__).resolve().parents[2]
log = logging.getLogger("live_view")
TRIAL_RE = re.compile(r"trial +\d+ seed +(\d+): (.*)")
VIEWPORT_CROP = "crop=1280:700:50:30"   # 3D viewport of the maximised Kit window at 1920x1080


def maximise_largest_window(display=b":1", w=1920, h=1080):
    """No window manager on the virtual display: resize the biggest top-level window (Kit) to the full screen."""
    x = C.CDLL("libX11.so.6")
    VP, UL, PUL = C.c_void_p, C.c_ulong, C.POINTER(C.c_ulong)
    x.XOpenDisplay.restype = VP
    x.XDefaultRootWindow.restype, x.XDefaultRootWindow.argtypes = UL, [VP]
    x.XQueryTree.argtypes = [VP, UL, PUL, PUL, C.POINTER(PUL), C.POINTER(C.c_uint)]
    x.XGetGeometry.argtypes = [VP, UL, PUL] + [C.POINTER(C.c_int)] * 2 + [C.POINTER(C.c_uint)] * 4
    x.XMoveResizeWindow.argtypes = [VP, UL, C.c_int, C.c_int, C.c_uint, C.c_uint]
    x.XSync.argtypes = [VP, C.c_int]
    d = x.XOpenDisplay(display)
    if not d:
        raise RuntimeError(f"cannot open X display {display!r}")
    r, p, kids, n = UL(), UL(), PUL(), C.c_uint()
    x.XQueryTree(d, x.XDefaultRootWindow(d), C.byref(r), C.byref(p), C.byref(kids), C.byref(n))
    best = None
    for i in range(n.value):
        rr, gx, gy = UL(), C.c_int(), C.c_int()
        ww, hh, bw, dp = C.c_uint(), C.c_uint(), C.c_uint(), C.c_uint()
        if x.XGetGeometry(d, kids[i], C.byref(rr), C.byref(gx), C.byref(gy), C.byref(ww), C.byref(hh),
                          C.byref(bw), C.byref(dp)) and (best is None or ww.value * hh.value > best[1]):
            best = (kids[i], ww.value * hh.value)
    if best:
        x.XMoveResizeWindow(d, best[0], 0, 0, w, h)
        x.XSync(d, 0)


class Grabber:
    """ffmpeg x11grab -> MJPEG on stdout; keeps the latest JPEG for any number of viewers."""

    def __init__(self, display, fps, width, view):
        import imageio_ffmpeg
        vf = ([VIEWPORT_CROP] if view == "viewport" else []) + [f"scale={width}:-2"]
        self.cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-f", "x11grab",
                    "-framerate", str(fps), "-video_size", "1920x1080", "-draw_mouse", "0", "-i", display,
                    "-vf", ",".join(vf), "-f", "mjpeg", "-q:v", "4", "pipe:1"]
        self.frame, self.seq, self.cv = b"", 0, threading.Condition()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE)
            buf = b""
            while chunk := proc.stdout.read(65536):
                buf += chunk
                while (s := buf.find(b"\xff\xd8")) >= 0 and (e := buf.find(b"\xff\xd9", s + 2)) >= 0:
                    with self.cv:
                        self.frame, self.seq = buf[s:e + 2], self.seq + 1
                        self.cv.notify_all()
                    buf = buf[e + 2:]
            log.warning("ffmpeg exited (%s), restarting", proc.wait())
            time.sleep(1)

    def wait(self, seq, timeout=5.0):
        with self.cv:
            self.cv.wait_for(lambda: self.seq != seq, timeout)
            return self.frame, self.seq


class Episodes:
    """One evaluate.py run (1 trial) at a time against the policy server; keeps the log tail and results."""

    def __init__(self, a):
        self.a, self.lock = a, threading.Lock()
        self.proc, self.seed, self.tail, self.results = None, None, [], []
        self.out = REPO / a.out
        self.out.mkdir(parents=True, exist_ok=True)

    def start(self, seed):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                return False
            self.seed, self.tail = seed, []
            logf = self.out / f"seed{seed}_{time.strftime('%H%M%S')}.log"
            cmd = [sys.executable, "-m", "aaltonomous.sim.evaluate", "--policy", f"ws://localhost:{self.a.host_port}",
                   "--team", self.a.team, "--task", self.a.task, "--seed", str(seed), "--trials", "1",
                   "--startup-timeout", "300", "--out", str(self.out / "eval")]
            self.proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            threading.Thread(target=self._follow, args=(self.proc, seed, logf), daemon=True).start()
            return True

    def _follow(self, proc, seed, logf):
        with open(logf, "w") as f:
            for line in proc.stdout:
                f.write(line)
                f.flush()
                line = line.rstrip()
                with self.lock:
                    if line.strip():
                        self.tail = (self.tail + [line])[-14:]
                    if m := TRIAL_RE.search(line):
                        self.results.append({"seed": int(m.group(1)), "line": m.group(2), "t": time.strftime("%H:%M")})
        proc.wait()

    def status(self):
        with self.lock:
            running = bool(self.proc and self.proc.poll() is None)
            return {"running": running, "seed": self.seed, "tail": self.tail, "results": self.results[::-1],
                    "policy": self.a.container, "next_seed": (max([r["seed"] for r in self.results], default=-1) + 1)}


PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Isaac Sim Live</title><style>
:root{--bg:#111316;--panel:#1b1e23;--fg:#e7e9ec;--mut:#8b929c;--acc:#76b900;--bad:#e5534b}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}
main{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:12px;padding:12px}
@media(max-width:900px){main{grid-template-columns:1fr}}
img{width:100%;border-radius:6px;background:#000;display:block}
.p{background:var(--panel);border-radius:6px;padding:12px;margin-bottom:12px}
h1{font-size:16px;margin:0 0 4px}h2{font-size:13px;color:var(--mut);margin:0 0 8px;text-transform:uppercase;letter-spacing:.04em}
button{background:var(--acc);color:#111;border:0;border-radius:4px;padding:8px 14px;font-weight:600;cursor:pointer}
button:disabled{opacity:.4;cursor:default}input{width:60px;background:#0d0f12;color:var(--fg);border:1px solid #333;border-radius:4px;padding:7px}
pre{white-space:pre-wrap;font-size:11px;color:var(--mut);margin:0;max-height:260px;overflow:auto}
.r{font:12px ui-monospace,monospace;padding:4px 0;border-bottom:1px solid #2a2e35}.live{color:var(--acc)}
</style></head><body><main><div><img src="/stream.mjpg" alt="Isaac Sim live"></div><div>
<div class="p"><h1>Isaac Sim, live</h1><div id="pol" style="color:var(--mut);font-size:12px"></div></div>
<div class="p"><h2>Episode</h2>seed <input id="seed" type="number" min="0" value="0"> <button id="go">Run episode</button>
<div id="st" style="margin-top:8px"></div></div>
<div class="p"><h2>Results</h2><div id="res"><span style="color:var(--mut)">none yet</span></div></div>
<div class="p"><h2>Evaluator log</h2><pre id="log"></pre></div></div></main><script>
let touched=false;const $=id=>document.getElementById(id);$('seed').oninput=()=>touched=true;
$('go').onclick=async()=>{await fetch('/run?seed='+$('seed').value,{method:'POST'});touched=false;poll()};
async function poll(){try{const s=await (await fetch('/status')).json();
$('pol').textContent='policy: '+s.policy+'  (sim.evaluate, 1 trial per run)';$('go').disabled=s.running;
$('st').innerHTML=s.running?'<span class="live">running seed '+s.seed+'</span>':'idle';
if(!touched&&!s.running)$('seed').value=s.next_seed;
$('res').innerHTML=s.results.length?s.results.map(r=>'<div class="r">'+r.t+'  seed '+r.seed+': '+r.line.replace(/</g,'&lt;')+'</div>').join(''):'<span style="color:var(--mut)">none yet</span>';
$('log').textContent=s.tail.join('\\n')}catch(e){}}
setInterval(poll,1500);poll();</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--container", required=True, help="submission image (kept running for the whole demo)")
    ap.add_argument("--team", default="demo")
    ap.add_argument("--task", default="sort_fixed", choices=["sort_fixed", "sort_language", "insert_shapes"])
    ap.add_argument("--host-port", type=int, default=8000, help="host port of the policy server")
    ap.add_argument("--display", default=":1")
    ap.add_argument("--view", default="full", choices=["full", "viewport"], help="whole Kit window or 3D viewport")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--width", type=int, default=1600, help="stream width (px)")
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--out", default="runs/live")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from aaltonomous.deploy.runner import Container
    maximise_largest_window(a.display.encode())
    grab, eps = Grabber(a.display, a.fps, a.width, a.view), Episodes(a)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if path == "/status":
                return self._send(200, json.dumps(eps.status()).encode(), "application/json")
            if path != "/stream.mjpg":
                return self._send(404, b"not found", "text/plain")
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seq = -1
            try:
                while True:
                    frame, seq = grab.wait(seq)
                    if frame:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            u = urlparse(self.path)
            if u.path != "/run":
                return self._send(404, b"not found", "text/plain")
            seed = int(parse_qs(u.query).get("seed", ["0"])[0])
            ok = eps.start(seed)
            self._send(200 if ok else 409, json.dumps({"started": ok}).encode(), "application/json")

    with Container(a.container, a.host_port):
        log.info("policy image %s on :%d; open http://localhost:%d", a.container, a.host_port, a.port)
        srv = ThreadingHTTPServer((a.bind, a.port), H)
        srv.daemon_threads = True
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            if eps.proc and eps.proc.poll() is None:
                eps.proc.terminate()


if __name__ == "__main__":
    main()
