"""Dials (neuromodulators) and knobs — global settings that retune every part.

A dial is a leaky integrator toward a target computed from this tick's signals:

    target = clamp(baseline + sum(gain_s * signal_s))
    D += rate * (target - D)

A knob is what the rest of the brain actually reads — a bounded linear map of
dials and drive urgencies:

    knob = clamp(base + sum(coeff_s * signal_s), min, max)

Signals are the flat dict from `events.summarize` plus `drive:<name>`,
`drives:max` and, for knobs, the dial levels by name.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .events import clamp


@dataclass
class Dial:
    name: str
    baseline: float = 0.5
    rate: float = 0.2
    gains: dict = field(default_factory=dict)
    level: float = 0.5

    def update(self, signals: dict[str, float]) -> None:
        target = self.baseline + sum(g * signals.get(s, 0.0) for s, g in self.gains.items())
        self.level += self.rate * (clamp(target) - self.level)


class DialSystem:
    """`clamps` fixes dials at a level whatever the signals — a drug, for
    experiments (e.g. {"noradrenaline": 0.9} for a stressed brain)."""

    def __init__(self, config: dict, levels: dict | None = None, clamps: dict | None = None):
        levels = levels or {}
        self.clamps = {k: float(v) for k, v in (clamps or {}).items()}
        unknown = set(self.clamps) - {n for n in config if not n.startswith("_")}
        if unknown:
            raise ValueError(f"cannot clamp unknown dial(s): {sorted(unknown)}")
        self.dials = []
        for name, spec in config.items():
            if name.startswith("_"):
                continue
            base = float(spec.get("baseline", 0.5))
            self.dials.append(Dial(
                name=name,
                baseline=base,
                rate=float(spec.get("rate", 0.2)),
                gains={k: float(v) for k, v in spec.get("gains", {}).items()
                       if not k.startswith("_")},
                level=self.clamps.get(name, float(levels.get(name, base))),
            ))

    def update(self, signals: dict[str, float]) -> None:
        for d in self.dials:
            if d.name in self.clamps:
                d.level = self.clamps[d.name]
            else:
                d.update(signals)

    def levels(self) -> dict[str, float]:
        return {d.name: d.level for d in self.dials}

    def snapshot(self) -> dict:
        return {d.name: round(d.level, 4) for d in self.dials}


_KNOB_RESERVED = {"base", "min", "max"}


def compute_knobs(config: dict, signals: dict[str, float]) -> dict[str, float]:
    out = {}
    for name, spec in config.items():
        if name.startswith("_"):
            continue
        v = float(spec.get("base", 0.0))
        for s, c in spec.items():
            if s in _KNOB_RESERVED or s.startswith("_"):
                continue
            v += float(c) * signals.get(s, 0.0)
        out[name] = clamp(v, float(spec.get("min", 0.0)), float(spec.get("max", 1.0)))
    return out
