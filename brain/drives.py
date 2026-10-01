"""Drives — the brainstem + hypothalamus. Needs declared in config, not code.

A drive is a need level L in [0,1] (0 = satisfied). Each tick:

    L -= decay * L                        (a leak toward 0, if declared: a state that fades, 3n frustration)
    L += pace * drift                     (drift_asleep instead while asleep, if declared)
    L += pace * rises[kind] * amount(e)   for each event e (pace 1 unless the brain entrains the drive)
    L -= satisfied_by[kind] * amount(e)
    L  = clamp(L)

A rule is either a number (gain per event) or {"per": field, "gain": g}
(gain per unit of e.data[field]). Urgency is how much the need is felt:
(L - setpoint) / (1 - setpoint), then damped by other drives' `inhibits`.
The most urgent drive is the current motive. `felt` bands turn a level into a
phrase for Laya; below the first band a drive is silent. A drive's `felt_gains` add signals (dial
levels) to the level for its phrase only: sleepiness felt from the pressure and the clock.
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
    drift_asleep: float | None = None                  # e.g. 0 for boredom: nobody gets bored asleep
    rises: dict = field(default_factory=dict)
    satisfied_by: dict = field(default_factory=dict)
    inhibits: dict = field(default_factory=dict)      # other drive -> damping weight
    felt: list = field(default_factory=list)           # [[level, phrase], ...] ascending
    enabled: bool = True
    escalates: bool = True                             # can deliberation help this need?
    level: float = 0.0
    pace: float = 1.0                                  # scales drift and rises (the brain's `entrain`, 3k)
    felt_gains: dict = field(default_factory=dict)     # signal -> gain added to the level for its phrase (3k)
    decay: float = 0.0                                 # a leak toward 0 each tick, x the level (3n: frustration fades)

    @classmethod
    def from_config(cls, name: str, spec: dict, level: float | None) -> "Drive":
        return cls(
            name=name,
            setpoint=float(spec.get("setpoint", 0.3)),
            drift=float(spec.get("drift", 0.0)),
            drift_asleep=float(spec["drift_asleep"]) if "drift_asleep" in spec else None,
            rises=_rules(spec.get("rises")),
            satisfied_by=_rules(spec.get("satisfied_by")),
            inhibits={k: float(v) for k, v in spec.get("inhibits", {}).items()
                      if not k.startswith("_")},
            felt=sorted(spec.get("felt", []), key=lambda b: b[0]),
            enabled=bool(spec.get("enabled", True)),
            escalates=bool(spec.get("escalates", True)),
            felt_gains={k: float(v) for k, v in spec.get("felt_gains", {}).items() if not k.startswith("_")},
            decay=float(spec.get("decay", 0.0)),
            level=float(spec.get("init", 0.0) if level is None else level),
        )

    def update(self, events, asleep: bool = False) -> None:
        L = self.level * (1.0 - self.decay)
        L += self.pace * (self.drift_asleep if asleep and self.drift_asleep is not None else self.drift)
        for e in events:
            if e.kind in self.rises:
                per, g = self.rises[e.kind]
                L += self.pace * g * _amount(e, per)
            if e.kind in self.satisfied_by:
                per, g = self.satisfied_by[e.kind]
                L -= g * _amount(e, per)
        self.level = clamp(L)

    def raw_urgency(self) -> float:
        if self.setpoint >= 1.0:
            return 0.0
        return clamp((self.level - self.setpoint) / (1.0 - self.setpoint))

    def phrase(self, signals: dict | None = None) -> str | None:
        """The felt band of the level, or of the level plus `felt_gains` x signals (3k: sleepiness is felt from
        the pressure and the circadian clock together, Dijk & Czeisler 1995); the need itself is unchanged."""
        lvl = self.level + sum(g * (signals or {}).get(s, 0.0) for s, g in self.felt_gains.items())
        out = None
        for threshold, text in self.felt:
            if lvl >= threshold:
                out = text
        return out


class DriveSystem:
    def __init__(self, config: dict, levels: dict | None = None):
        levels = levels or {}
        self.drives = [Drive.from_config(n, s, levels.get(n))
                       for n, s in config.items() if not n.startswith("_")]
        self.active = [d for d in self.drives if d.enabled]
        self._urg: dict[str, float] = {}
        self.felt_signals: dict[str, float] = {}           # what `felt_gains` read (the dials, set by the brain)
        self._recompute()

    def update(self, events, asleep: bool = False) -> None:
        for d in self.active:
            d.update(events, asleep)
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

    def set_pace(self, name: str, pace: float) -> None:
        for d in self.active:
            if d.name == name:
                d.pace = pace

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
        return [p for d in ranked if (p := d.phrase(self.felt_signals))]

    def signals(self) -> dict[str, float]:
        out = {f"drive:{n}": u for n, u in self._urg.items()}
        # the need itself, felt or not yet (step 10aa: ghrelin rises with the deficit before hunger is felt)
        out.update({f"level:{d.name}": d.level for d in self.active})
        out["drives:max"] = max(self._urg.values(), default=0.0)
        return out

    def snapshot(self) -> dict:
        return {d.name: [round(d.level, 4), round(self._urg.get(d.name, 0.0), 4)]
                for d in self.active}

    def levels(self) -> dict[str, float]:
        return {d.name: d.level for d in self.drives}
