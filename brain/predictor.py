"""The cerebellum: a forward model that learns online — "doing X here leads to Y".

Tabular on purpose: a context is the exact situation text + the action, and it
keeps counts of the outcomes that followed. Before acting it predicts; after
acting it observes and learns. Per context it tracks an error average, which
gives the three signals the rest of the brain runs on:

- **error**     1 − P(what happened) under the counts before this observation
                (1 for a context never seen before);
- **progress**  how much the context's error average just fell — learning
                progress. A context that becomes predictable yields progress for a
                while, then none; **pure noise never does** (its error stays 1),
                which is what defuses the noisy-TV trap when progress, not
                novelty, relieves boredom;
- **surprise**  how much more wrong it was than it expected to be (error minus
                the context's error average, floored at 0) — the noradrenaline
                signal. A well-learned context that suddenly breaks gives a lot.

Plus **uncertainty** (the context's error average before acting), the expected
uncertainty acetylcholine tracks. Persisted in the bundle (`predictor.json`):
the cerebellum is long-term, it survives sleep and shutdown.

**Place verdicts.** A context that never repeats (random glyphs) never gets an
expectation, so System 2 used to hear nothing about it and read patterns into
noise (apophenia, step 5). A coarser table keyed by the *place* (the situation's
first sentence, "I am at the sign") + action counts distinct outcomes; when
nearly every try gave something new, System 2 is told so ("different every
time so far"). A verdict, not an explanation: content (pages of a text) and
noise look the same to it.

**Others (step 8b).** In a shared world the cerebellum also learns how another
mind responds to what this one does: context "with <someone with an amber
mark>" + my action, outcome the *kind* of response one tick later ("they
answered in words", "they walked away", "no sign from them"...). Kinds, not
words: exact words never repeat (like the sign), but whether the other one
answers does, so getting to know someone is learning progress.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

SOMEONE = re.compile(r"^(Someone with an? [\w-]+ mark) is here(, lying still)?")


def others_here(percepts) -> list[str]:
    """The awake others this mind sees right now ("Someone with an amber mark")."""
    return [m.group(1) for p in percepts if (m := SOMEONE.match(p)) and not m.group(2)]


def social_response(who: str, percepts, did: str | None = None) -> str:
    """The kind of response `who` gave, from what this mind senses one tick later."""
    mine = [p for p in percepts if p.startswith(who)]
    if did == "walk":                               # I moved on: did they come along?
        return "they came along" if mine else "I left them behind"
    if any(' said: "' in p for p in mine):
        return "they answered in words"
    if any("called out" in p for p in mine):
        return "they called out"
    if any("walked on" in p for p in mine):
        return "they walked away"
    if any(" is here, lying still" in p for p in mine):
        return "they fell asleep"
    if any(" is here" not in p for p in mine):
        return "they did something here"
    if mine:
        return "no sign from them"
    return "they were gone"


class Predictor:
    def __init__(self, path, rate: float = 0.3, max_contexts: int = 5000,
                 unpredictable_after: int = 4):
        self.path = Path(path)
        self.rate = rate
        self.max_contexts = max_contexts
        self.unpredictable_after = unpredictable_after
        self.contexts: dict[str, dict] = {}
        self.places: dict[str, dict] = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                saved = json.load(f)
            self.contexts = saved.get("contexts", {})
            self.places = saved.get("places", {})

    @staticmethod
    def place(situation: str) -> str:
        return situation.split(". ")[0].strip()

    def verdict(self, situation: str, action: str) -> int | None:
        """Tries so far if this action here has given something new nearly every time."""
        p = self.places.get(self.key(self.place(situation), action))
        if p and p["n"] >= self.unpredictable_after and p["distinct"] >= 0.9 * p["n"]:
            return p["n"]
        return None

    @staticmethod
    def key(situation: str, action: str) -> str:
        return f"{situation} || {action}"

    def predict(self, situation: str, action: str) -> tuple[str | None, float, float]:
        """(most likely outcome or None, its probability, expected error) — before acting."""
        c = self.contexts.get(self.key(situation, action))
        if not c:
            return None, 0.0, 1.0
        best = max(c["counts"], key=c["counts"].get)
        return best, c["counts"][best] / (c["n"] + 1), c["err"]

    def expectations(self, situation: str, verbs) -> list[str]:
        """What the mind expects each action to do here, for deliberation."""
        out = []
        for v in verbs:
            best, p, _ = self.predict(situation, v)
            if best:
                out.append(f"{v}: {best} ({p:.0%} sure)")
            elif (n := self.verdict(situation, v)):
                out.append(f"{v}: different every time so far ({n} tries here), no outcome has repeated")
        return out

    def social_expectations(self, whos, verbs) -> list[str]:
        """What the mind has learned to expect from each other one present, per action."""
        out = []
        for who in whos:
            for v in verbs:
                best, p, _ = self.predict(f"with {who}", v)
                if best:
                    out.append(f"{v}, with {who[0].lower() + who[1:]} here: {best} ({p:.0%} sure)")
        return out

    def observe(self, situation: str, action: str, outcome: str) -> dict:
        k = self.key(situation, action)
        c = self.contexts.get(k)
        if c is None:
            if len(self.contexts) >= self.max_contexts:
                return {"error": 1.0, "progress": 0.0, "surprise": 0.0, "uncertainty": 1.0, "new": True}
            c = self.contexts[k] = {"counts": {}, "n": 0, "err": 1.0}
            error, before = 1.0, 1.0
        else:
            before = c["err"]
            # n + 1 in the denominator keeps room for an outcome never seen here
            error = 1.0 - c["counts"].get(outcome, 0) / (c["n"] + 1)
        pk = self.key(self.place(situation), action)
        pl = self.places.setdefault(pk, {"n": 0, "distinct": 0, "seen": []})
        pl["n"] += 1
        h = outcome[:120]
        if h not in pl["seen"]:
            pl["distinct"] += 1
            if len(pl["seen"]) < 40:
                pl["seen"].append(h)
        c["counts"][outcome] = c["counts"].get(outcome, 0) + 1
        c["n"] += 1
        c["err"] = before + self.rate * (error - before)
        return {
            "error": round(error, 4),
            "progress": round(max(0.0, before - c["err"]), 4),
            "surprise": round(max(0.0, error - before), 4),
            "uncertainty": round(before, 4),
            "new": c["n"] == 1,
        }

    def save(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"rate": self.rate, "contexts": self.contexts, "places": self.places}, f,
                      ensure_ascii=False)
        os.replace(tmp, self.path)
