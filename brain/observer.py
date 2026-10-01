"""The observer: a one-way mirror onto running brains and their worlds.

    python -m brain.observer --host <private-ip> --port 8700 [--root .] [--read-only]

A small stdlib HTTP server and one page (`observer.html`). It READS the run logs
(`runs/*.jsonl`) and the bundles' wikis, and its only write is the pause flag a
run polls (`runs/<name>.control.json`), which pauses the brain and its world
together. Nothing a brain can sense comes from here: watching stays outside the
world. `--read-only` removes the pause button (for friends, later).

Endpoints: GET /api/runs, /api/frames?run=&since=&limit=, /api/wiki?bundle=,
/api/digest?run=&from=&to=; POST /api/control?run= {"paused": bool}.
Logs are read incrementally and kept as trimmed frames, so a world-day of
86,400 ticks is not re-read on every poll.

Shared worlds (step 8, `together.py`): each brain has its own log, and the
meta names the world's one switch (`control`), so pausing any of them pauses
the whole world. Frames carry who was near (`with`), what the brain said and
what it heard; `/api/frames` adds where the others are now (`others`).

Lands (step 9): frames carry the body's position; `/api/world?run=` serves the
land's current map from `worlds/<name>/world.json` (terrain, items lying, where
everyone stands, day or night) for the page's tile map. Replays show positions
over today's terrain: the terrain's own history is not kept yet.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .digest import clock, digest_rows, place

NAME = re.compile(r"^[\w.-]{1,80}$")
PAGE = Path(__file__).with_name("observer.html")


def trim(r: dict) -> dict:
    """What the page needs from a log row (a few hundred bytes instead of ~3 KB)."""
    th = r.get("thought")
    if th:
        th = {k: th.get(k) for k in ("goal", "action", "expectation", "expectation_met", "error", "seconds")}
    c = r.get("consolidation")
    if c:
        c = {k: c.get(k) for k in ("notes", "created", "updated", "error")}
    heard = [p[:300] for p in r.get("percepts", [])
             if ' said: "' in p or "called out" in p or p.startswith("I heard")]
    return {
        "t": r.get("t"), "asleep": bool(r.get("asleep")), "place": place(r),
        "with": r.get("with"), "said": r.get("said"), "heard": heard or None,
        "pos": r.get("pos"), "facing": r.get("facing"), "held": r.get("held"),
        "action": r.get("action"), "goal": r.get("goal"), "feeling": r.get("feeling"),
        "drives": r.get("drives", {}), "thought": th, "consolidation": c,
        "outcomes": [o[:300] for o in r.get("outcomes", [])],
        "escalated": r.get("escalated"), "distress": r.get("distress"),
        "distress_warn": r.get("distress_warn"), "distress_pause": r.get("distress_pause"),
        "progress": (r.get("prediction") or {}).get("progress"),
    }


class Run:
    """One log file, read from where the last read stopped."""

    def __init__(self, path: Path):
        self.path = path
        self.offset = 0
        self.frames: list[dict] = []
        self.meta: dict = {}
        self.controls: list[dict] = []
        self.lock = threading.Lock()

    def refresh(self) -> None:
        with self.lock:
            size = self.path.stat().st_size
            if size < self.offset:                       # truncated / rewritten: start over
                self.offset, self.frames, self.controls = 0, [], []
            if size == self.offset:
                return
            with open(self.path, "rb") as f:
                f.seek(self.offset)
                data = f.read()
            end = data.rfind(b"\n") + 1                   # only whole lines
            for line in data[:end].splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if "meta" in r:
                    self.meta = r["meta"]
                elif "control" in r:
                    self.controls.append(r["control"])
                else:
                    self.frames.append(trim(r))
            self.offset += end


class Observer:
    def __init__(self, root: Path, read_only: bool):
        self.root = root
        self.read_only = read_only
        self.runs: dict[str, Run] = {}

    def run(self, name: str) -> Run | None:
        if not NAME.match(name or ""):
            return None
        p = self.root / "runs" / f"{name}.jsonl"
        if not p.is_file():
            return None
        r = self.runs.setdefault(name, Run(p))
        r.refresh()
        return r

    def control_path(self, name: str) -> Path:
        r = self.runs.get(name)
        shared = r.meta.get("control") if r else None          # a shared world: one switch for all
        if shared and NAME.match(shared):
            return self.root / "runs" / shared
        return self.root / "runs" / f"{name}.control.json"

    def others(self, name: str) -> dict:
        """Where the other minds of a shared world are, from their own logs."""
        r = self.runs.get(name)
        if not r or not r.meta.get("together"):
            return {}
        out = {}
        for other, mark in (r.meta.get("others") or {}).items():
            o = self.run(f"{r.meta['together']}.{other}")
            if o and o.frames:
                f = o.frames[-1]
                out[other] = {"mark": mark, "place": f["place"], "asleep": f["asleep"], "t": f["t"]}
        return out

    def control(self, name: str) -> dict:
        try:
            return json.loads(self.control_path(name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def list_runs(self) -> list[dict]:
        out = []
        for p in sorted((self.root / "runs").glob("*.jsonl"), key=lambda p: -p.stat().st_mtime):
            r = self.run(p.stem)
            if not r:
                continue
            last = r.frames[-1]["t"] if r.frames else None
            age = time.time() - p.stat().st_mtime
            ctl = self.control(p.stem)
            out.append({"name": p.stem, "bundle": r.meta.get("bundle"), "world": r.meta.get("world"),
                        "together": r.meta.get("together"), "mark": r.meta.get("mark"),
                        "started": r.meta.get("started"), "t": last,
                        "clock": clock(last) if last is not None else None,
                        "live": age < 15 or bool(ctl.get("paused") and age < 3600),
                        "paused": bool(ctl.get("paused")), "paused_by": ctl.get("by"),
                        "reason": ctl.get("reason")})
        return out

    def land(self, name: str) -> dict:
        r = self.run(name)
        t = r.meta.get("together") if r else None
        if not t or not NAME.match(t):
            return {}
        try:
            st = json.loads((self.root / "worlds" / t / "world.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if "rows" not in st:
            return {}
        return {k: st.get(k) for k in ("size", "rows", "legend", "items", "phase", "clock",
                                       "shelter", "orchard", "bodies")}

    def wiki(self, bundle: str) -> list[dict]:
        if not NAME.match(bundle or ""):
            return []
        d = self.root / "bundles" / bundle / "wiki"
        notes = []
        for p in sorted(d.glob("*.md")) if d.is_dir() else []:
            if p.name == "index.md":
                continue
            text = p.read_text(encoding="utf-8")
            meta, body = {}, text
            m = re.match(r"^---\n(.*?)\n---\n?(.*)$", text, re.S)
            if m:
                body = m.group(2)
                for line in m.group(1).splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        meta[k.strip()] = v.strip().strip('"')
            notes.append({"title": meta.get("title", p.stem), "updated_t": meta.get("updated_t"),
                          "text": body.strip()})
        return sorted(notes, key=lambda n: -int(n["updated_t"] or 0))


def make_handler(obs: Observer):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code: int, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype + ("; charset=utf-8" if "json" in ctype or "html" in ctype else ""))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path == "/":
                return self._send(200, PAGE.read_bytes(), "text/html")
            if u.path == "/api/runs":
                return self._send(200, {"runs": obs.list_runs(), "read_only": obs.read_only})
            if u.path == "/api/frames":
                r = obs.run(q.get("run", ""))
                if not r:
                    return self._send(404, {"error": "no such run"})
                since = int(q.get("since", -1))
                limit = min(int(q.get("limit", 2000)), 20000)
                fr = [f for f in r.frames if f["t"] is not None and f["t"] > since][:limit]
                return self._send(200, {"meta": r.meta, "frames": fr, "controls": r.controls,
                                        "control": obs.control(q["run"]), "others": obs.others(q["run"]),
                                        "last_t": r.frames[-1]["t"] if r.frames else None})
            if u.path == "/api/world":
                return self._send(200, obs.land(q.get("run", "")))
            if u.path == "/api/wiki":
                return self._send(200, {"notes": obs.wiki(q.get("bundle", ""))})
            if u.path == "/api/digest":
                r = obs.run(q.get("run", ""))
                if not r:
                    return self._send(404, {"error": "no such run"})
                a, b = int(q.get("from", 0)), int(q.get("to", 10**12))
                rows = [f for f in r.frames if a <= f["t"] < b]
                ctl = [c for c in r.controls if a <= c.get("t", 0) < b]
                return self._send(200, digest_rows(rows, ctl))
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path != "/api/control":
                return self._send(404, {"error": "not found"})
            if obs.read_only:
                return self._send(403, {"error": "read-only observer"})
            name = q.get("run", "")
            if not obs.run(name):
                return self._send(404, {"error": "no such run"})
            n = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(min(n, 4096)) or b"{}")
            except ValueError:
                return self._send(400, {"error": "bad json"})
            state = {"paused": bool(body.get("paused")), "by": "creator",
                     "reason": "paused from the observer" if body.get("paused") else "resumed"}
            p = obs.control_path(name)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(state), encoding="utf-8")
            tmp.replace(p)
            return self._send(200, state)
    return H


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1", help="bind address (a private IP; never 0.0.0.0)")
    ap.add_argument("--port", type=int, default=8700)
    ap.add_argument("--root", default=".", help="the folder holding runs/ and bundles/")
    ap.add_argument("--read-only", action="store_true", help="no pause button (for guests)")
    a = ap.parse_args(argv)
    if a.host in ("0.0.0.0", "::"):
        print("refusing to bind every interface: pass a private IP", file=sys.stderr)
        return 2
    obs = Observer(Path(a.root).expanduser().resolve(), a.read_only)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(obs))
    print(f"observer on http://{a.host}:{a.port}/ (root {obs.root}{', read-only' if a.read_only else ''})",
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
