"""Drives — the brainstem + hypothalamus. Needs declared in config, not code.

A drive is a need level L in [0,1] (0 = satisfied). Each tick:

    L += drift
    L += rises[kind] * amount(e)          for each event e
    L -= satisfied_by[kind] * amount(e)
    L  = clamp(L)

A rule is either a number (gain per event) or {"per": field, "gain": g}
(gain per unit of e.data[field]). Urgency is how much the need is felt:
(L - setpoint) / (1 - setpoint), then damped by other drives' `inhibits`.
The most urgent drive is the current motive. `felt` bands turn a level into a
phrase for Laya; below the first band a drive is silent.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .events import clamp


def _rules(spec: dict) -> dict[str, tuple[str | None, float]]:
    out = {}
    for kind, rule in (spec or {}).items():
        if kind.startswith("_"):
            continue
        if isinstance(rule, dict):
            out[kind] = (rule.get("per"), float(rule["gain"]))
        else:
            out[kind] = (None, float(rule))
    return out


def _amount(e, per: str | None) -> float:
    if per is None:
        return 1.0
    v = e.data.get(per, 0.0)
    return float(v) if isinstance(v, (int, float)) else 0.0


@dataclass
class Drive:
    name: str
    setpoint: float = 0.3
    drift: float = 0.0
    rises: dict = field(default_factory=dict)
    satisfied_by: dict = field(default_factory=dict)
    inhibits: dict = field(default_factory=dict)      # other drive -> damping weight
    felt: list = field(default_factory=list)           # [[level, phrase], ...] ascending
    enabled: bool = True
    escalates: bool = True                             # can deliberation help this need?
    level: float = 0.0

    @classmethod
    def from_config(cls, name: str, spec: dict, level: float | None) -> "Drive":
        return cls(
            name=name,
            setpoint=float(spec.get("setpoint", 0.3)),
            drift=float(spec.get("drift", 0.0)),
            rises=_rules(spec.get("rises")),
            satisfied_by=_rules(spec.get("satisfied_by")),
            inhibits={k: float(v) for k, v in spec.get("inhibits", {}).items()
                      if not k.startswith("_")},
            felt=sorted(spec.get("felt", []), key=lambda b: b[0]),
            enabled=bool(spec.get("enabled", True)),
            escalates=bool(spec.get("escalates", True)),
            level=float(spec.get("init", 0.0) if level is None else level),
        )

    def update(self, events) -> None:
        L = self.level + self.drift
        for e in events:
            if e.kind in self.rises:
                per, g = self.rises[e.kind]
                L += g * _amount(e, per)
            if e.kind in self.satisfied_by:
                per, g = self.satisfied_by[e.kind]
                L -= g * _amount(e, per)
        self.level = clamp(L)

    def raw_urgency(self) -> float:
        if self.setpoint >= 1.0:
            return 0.0
        return clamp((self.level - self.setpoint) / (1.0 - self.setpoint))

    def phrase(self) -> str | None:
        out = None
        for threshold, text in self.felt:
            if self.level >= threshold:
                out = text
        return out


class DriveSystem:
    def __init__(self, config: dict, levels: dict | None = None):
        levels = levels or {}
        self.drives = [Drive.from_config(n, s, levels.get(n))
                       for n, s in config.items() if not n.startswith("_")]
        self.active = [d for d in self.drives if d.enabled]
        self._urg: dict[str, float] = {}
        self._recompute()

    def update(self, events) -> None:
        for d in self.active:
            d.update(events)
        self._recompute()

    def _recompute(self) -> None:
        raw = {d.name: d.raw_urgency() for d in self.active}
        urg = {}
        for d in self.active:
            u = raw[d.name]
            for other in self.active:
                w = other.inhibits.get(d.name)
                if w:
                    u *= 1.0 - w * raw[other.name]
            urg[d.name] = clamp(u)
        self._urg = urg

    def level(self, name: str) -> float:
        return next((d.level for d in self.active if d.name == name), 0.0)

    def urgency(self, name: str) -> float:
        return self._urg.get(name, 0.0)

    def urgencies(self) -> dict[str, float]:
        return dict(self._urg)

    def motive(self) -> str | None:
        if not self._urg:
            return None
        name, u = max(self._urg.items(), key=lambda kv: kv[1])
        return name if u > 0.0 else None

    def escalating_motive(self) -> str | None:
        """The most urgent need that deliberation could help with (not, e.g.,
        compute hunger, which thinking only makes worse)."""
        cands = {d.name: self._urg.get(d.name, 0.0) for d in self.active if d.escalates}
        if not cands:
            return None
        name, u = max(cands.items(), key=lambda kv: kv[1])
        return name if u > 0.0 else None

    def felt(self) -> list[str]:
        """Phrases for every drive above its first band, most urgent first."""
        ranked = sorted(self.active, key=lambda d: -self._urg.get(d.name, 0.0))
        return [p for d in ranked if (p := d.phrase())]

    def signals(self) -> dict[str, float]:
        out = {f"drive:{n}": u for n, u in self._urg.items()}
        out["drives:max"] = max(self._urg.values(), default=0.0)
        return out

    def snapshot(self) -> dict:
        return {d.name: [round(d.level, 4), round(self._urg.get(d.name, 0.0), 4)]
                for d in self.active}

    def levels(self) -> dict[str, float]:
        return {d.name: d.level for d in self.drives}
