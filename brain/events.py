"""The event stream: the one bus every part of the brain publishes to.

A tick's events are the only input drives and dials see, which is what lets
them be declared in config (reducers over event kinds) instead of written in
code, and what makes a run replayable from its log.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class Event:
    t: int
    src: str
    kind: str
    data: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


def event(src: str, kind: str, **data) -> Event:
    """Build an event outside the brain (e.g. in a world); the stream stamps `t`."""
    return Event(0, src, kind, data)


class Stream:
    """Per-tick buffer. Parts `emit`/`add`; the brain `drain`s at the end of a tick."""

    def __init__(self, t: int):
        self.t = t
        self._buf: list[Event] = []

    def emit(self, src: str, kind: str, **data) -> Event:
        e = Event(self.t, src, kind, data)
        self._buf.append(e)
        return e

    def add(self, events) -> list[Event]:
        for e in events:
            e.t = self.t
            self._buf.append(e)
        return list(events)

    def drain(self) -> list[Event]:
        out, self._buf = self._buf, []
        return out


def summarize(events) -> dict[str, float]:
    """Flatten a tick's events into signals: `kind` -> count, `kind.field` -> sum
    of every numeric field. Drives, dials and knobs all read this one dict."""
    out: dict[str, float] = {}
    for e in events:
        out[e.kind] = out.get(e.kind, 0.0) + 1.0
        for k, v in e.data.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                key = f"{e.kind}.{k}"
                out[key] = out.get(key, 0.0) + float(v)
    return out


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x
