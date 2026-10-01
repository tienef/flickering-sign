"""Optional: hold a GPU lease from an arbiter daemon (boxctl-style HTTP API) while a run uses the card.

Only used when the genome's `middle.lease` names a model and BOXCTL_URL points at such a daemon; without one,
leave `middle.lease` null and run your own vLLM server. What follows describes the API it speaks.

The daemon is the single arbiter of a mono-tenant card. Its `free`
mode means "the card is deliberately empty -- the owner wants it for a game or
maintenance"; used for a GPU session. Taking a `free` lease makes
boxctl itself evict the resident text engine (vLLM/q27) -- honouring its "boxctl
is the ONLY actor that stops/starts vLLM" invariant -- and hold the card empty
against idle-restore until we release, when the boot default comes back.

Use as a context manager around a GPU run:

    with gpu_lease(hold="brain"):
        backend = LayaBackend(device="cuda")
        ...run the sim...

Stdlib only (urllib). Auth: set BOXCTL_TOKEN if the daemon requires a bearer.
This module never touches the card directly -- it only asks boxctl.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from contextlib import contextmanager

BOXCTL = os.environ.get("BOXCTL_URL", "")


def _resolve_token() -> str:
    """The daemon's bearer: env first, else a user-readable file.

    boxctl's mutating endpoints need BOXCTL_TOKEN (reads are open). The secret
    lives in a root-only file; rather than sudo on every run, a
    one-time `sudo` drops it into .boxctl.env (user-owned, chmod
    600) as a `BOXCTL_TOKEN=...` line, which this reads. Override the path with
    BOXCTL_ENV_FILE.
    """
    tok = os.environ.get("BOXCTL_TOKEN")
    if tok:
        return tok
    path = os.environ.get("BOXCTL_ENV_FILE",
                          ".boxctl.env")
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("BOXCTL_TOKEN="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return ""


TOKEN = _resolve_token()


def _post(path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BOXCTL + path, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if TOKEN:
        req.add_header("Authorization", f"Bearer {TOKEN}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read() or "{}")


def _get(path):
    req = urllib.request.Request(BOXCTL + path)
    if TOKEN:
        req.add_header("Authorization", f"Bearer {TOKEN}")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read() or "{}")


class TextLease:
    """A boxctl `text` lease on one model (three levels, P1 step 6; owner 2026-09-30: the brain reaches its model
    on the engine's own port under a lease, not through a gateway alias). boxctl loads the model if another is
    resident (a vLLM swap: ~6 min; other consumers queue behind it), holds it while heartbeated, and on release
    its idle reconciler brings the boot default back. `acquire` waits until the engine serves `served_id` at
    `serve_url`; `release` is idempotent (runners register it with atexit; the TTL reclaims a killed run's)."""

    def __init__(self, model: str, hold: str = "brain", cls: str = "batch", ttl: int = 900,
                 serve_url: str | None = None, served_id: str | None = None, poll_timeout: int = 1500,
                 verbose: bool = True):
        self.model, self.hold, self.cls, self.ttl = model, hold, cls, ttl
        self.serve_url, self.served_id = serve_url, served_id or model
        self.poll_timeout, self.verbose = poll_timeout, verbose
        self.lease = None
        self._stop = threading.Event()

    def _say(self, *a):
        if self.verbose:
            print("[lease]", *a, flush=True)

    def served(self) -> list[str]:
        try:
            with urllib.request.urlopen(self.serve_url.rstrip("/") + "/models", timeout=10) as r:
                return [m["id"] for m in json.loads(r.read())["data"]]
        except Exception:  # noqa: BLE001 - the engine is starting or gone
            return []

    def acquire(self) -> "TextLease":
        t0 = time.time()
        res = _post("/v1/gpu/lease", {"class": self.cls, "mode": "text", "model": self.model, "hold": self.hold,
                                      "ttl": self.ttl})
        self.lease = res["lease"]
        self._say(f"requested {self.lease} (text {self.model}/{self.cls}); state={res.get('state')}")
        while res.get("state") != "granted":
            if res.get("state") in ("preempted", "denied", "error"):
                raise RuntimeError(f"lease not granted: {res}")
            if time.time() - t0 > self.poll_timeout:
                self.release()
                raise TimeoutError(f"lease still {res.get('state')} after {self.poll_timeout}s")
            time.sleep(5)
            res = _get(f"/v1/gpu/lease/{self.lease}")
        threading.Thread(target=self._beat, daemon=True).start()
        while self.serve_url and self.served_id not in self.served():
            if time.time() - t0 > self.poll_timeout:
                self.release()
                raise TimeoutError(f"{self.served_id} not served at {self.serve_url} after {self.poll_timeout}s")
            time.sleep(5)
        self._say(f"granted and serving in {time.time() - t0:.0f} s")
        return self

    def _beat(self):
        while not self._stop.wait(max(15, self.ttl // 3)):
            try:
                _post(f"/v1/gpu/lease/{self.lease}/heartbeat")
            except Exception as e:  # noqa: BLE001 - best-effort; the TTL is the backstop
                self._say(f"heartbeat failed: {e}")

    def release(self) -> None:
        if self.lease is None:
            return
        self._stop.set()
        lease, self.lease = self.lease, None
        try:
            _post("/v1/gpu/release", {"lease": lease})
            self._say(f"released {lease}")
        except Exception as e:  # noqa: BLE001
            self._say(f"release failed (the TTL will reclaim it): {e}")


def text_lease_for(middle_cfg: dict, name: str):
    """The runners' helper: a held TextLease when the genome's `middle` names one (`lease`: a boxctl key), else
    None. Released at exit."""
    key = (middle_cfg or {}).get("lease")
    if not key:
        return None
    import atexit
    lease = TextLease(key, hold=f"brain:{name}", serve_url=middle_cfg.get("url"),
                      served_id=middle_cfg.get("model")).acquire()
    atexit.register(lease.release)
    return lease


@contextmanager
def gpu_lease(hold="brain", cls="batch", ttl=600, poll_timeout=420, verbose=True):
    """Hold a `free` lease for the duration of the block.

    cls: "batch" (a live interactive session preempts the sim -- polite) or
         "interactive" (the sim is protected, like "reserve for games").
    ttl: seconds; a background thread heartbeats at ttl/2 so long runs don't lapse.
    """
    res = _post("/v1/gpu/lease",
                {"class": cls, "mode": "free", "hold": hold, "ttl": ttl})
    lease = res["lease"]
    if verbose:
        print(f"[lease] requested {lease} (free/{cls}); state={res.get('state')}")

    # wait for the card to actually be free (boxctl evicting vLLM can take a bit)
    deadline = time.time() + poll_timeout
    while res.get("state") != "granted":
        if res.get("state") in ("preempted", "denied", "error"):
            raise RuntimeError(f"lease not granted: {res}")
        if time.time() > deadline:
            _post("/v1/gpu/release", {"lease": lease})
            raise TimeoutError(f"lease still {res.get('state')} after {poll_timeout}s")
        time.sleep(3)
        res = _get(f"/v1/gpu/lease/{lease}")
    if verbose:
        print(f"[lease] granted {lease} — card is free for GPU-Laya")

    stop = threading.Event()

    def _beat():
        while not stop.wait(max(15, ttl // 2)):
            try:
                _post(f"/v1/gpu/lease/{lease}/heartbeat")
            except Exception as e:  # noqa: BLE001 - heartbeat is best-effort
                print(f"[lease] heartbeat failed: {e}")

    hb = threading.Thread(target=_beat, daemon=True)
    hb.start()
    try:
        yield lease
    finally:
        stop.set()
        try:
            _post("/v1/gpu/release", {"lease": lease})
            if verbose:
                print(f"[lease] released {lease} — boot default returns")
        except Exception as e:  # noqa: BLE001
            print(f"[lease] release failed (TTL backstop will reclaim): {e}")
