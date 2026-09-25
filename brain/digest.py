"""Digests: what happened in a window of world time, in plain words.

    python -m brain.digest runs/ivy.jsonl [--from T] [--to T]

Used three ways: `run.py` writes one every `loop.digest_every` ticks (a world
day) next to the log, the observer computes one for any window, and by hand.
Works on full log rows or on the observer's trimmed frames (the same keys).
One tick is one second of world time (`loop.seconds_per_tick`).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

_PLACE = re.compile(r"I am at the (\w+)")


def clock(t: int) -> str:
    """World time for tick t: 'day 1, 03:25:07' (day 1 starts at t=0)."""
    d, rem = divmod(int(t), 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    return f"day {d + 1}, {h:02d}:{m:02d}:{s:02d}"


def place(row: dict) -> str | None:
    if row.get("place"):
        return row["place"]
    for p in row.get("percepts", []):
        m = _PLACE.match(p)
        if m:
            return m.group(1)
    return None


def load(path: Path) -> tuple[list[dict], list[dict]]:
    rows, controls = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if "control" in r:
                    controls.append(r["control"])
                elif "meta" not in r:
                    rows.append(r)
    return rows, controls


def digest_rows(rows: list[dict], controls: list[dict] | None = None) -> dict:
    if not rows:
        return {"ticks": 0}
    awake = [r for r in rows if not r.get("asleep")]
    places = Counter(p for r in awake if (p := place(r)))
    n = max(1, sum(places.values()))
    top = Counter()
    for r in awake:
        d = r.get("distress")
        if d:
            top[d["drive"]] += 1
    thoughts = [r["thought"] for r in rows if r.get("thought") and not r["thought"].get("error")]
    notes, created, revised = [], 0, 0
    for r in rows:
        c = r.get("consolidation")
        if c and not c.get("error"):
            created += c.get("created", 0)
            revised += c.get("updated", 0)
            notes += [t for t in c.get("notes", []) if t and t not in notes]
    reads = [o for r in awake for o in r.get("outcomes", []) if o.startswith("I read '")]
    papers = []
    for o in reads:
        m = re.match(r"I read '(.+?)', page", o)
        if m and m.group(1) not in papers:
            papers.append(m.group(1))
    return {
        "from": rows[0]["t"], "to": rows[-1]["t"], "ticks": len(rows),
        "awake": len(awake),
        "sleeps": sum(1 for a, b in zip(rows, rows[1:]) if b.get("asleep") and not a.get("asleep")),
        "thoughts": len(thoughts),
        "places": {p: round(k / n, 2) for p, k in places.most_common()},
        "actions": dict(Counter(r.get("action") for r in awake).most_common()),
        "worst_band_ticks": dict(top),
        "guard_pauses": [c for c in (controls or []) if c.get("paused") and c.get("by") == "guard"],
        "pauses": sum(1 for c in (controls or []) if c.get("paused")),
        "notes_touched": notes, "notes_created": created, "notes_revised": revised,
        "pages_read": len(reads), "papers": papers,
        "first_goals": [t.get("goal") for t in thoughts[:3]],
        "last_goals": [t.get("goal") for t in thoughts[-3:]],
        "end_drives": rows[-1].get("drives", {}),
    }


def to_markdown(d: dict, title: str) -> str:
    if not d.get("ticks"):
        return f"# {title}\n\nNothing happened in this window.\n"
    L = [f"# {title}", "",
         f"*{clock(d["from"])} to {clock(d["to"])}: {d['ticks']} ticks, {d['awake']} awake, "
         f"{d['sleeps']} sleeps, {d['thoughts']} thoughts.*", ""]
    L += ["## Where it spent its time", ""]
    L += [f"- {p}: {int(s * 100)}%" for p, s in d["places"].items()] or ["- nowhere (asleep)"]
    L += ["", "## What it learned", ""]
    if d["notes_touched"]:
        L.append(f"{d['notes_created']} notes created, {d['notes_revised']} revised: "
                 + ", ".join(d["notes_touched"]) + ".")
    else:
        L.append("No consolidation in this window.")
    if d["pages_read"]:
        L += ["", f"It read {d['pages_read']} pages at the archive: " + "; ".join(d["papers"]) + "."]
    L += ["", "## What it was aiming for", ""]
    L += [f"- first: {g}" for g in d["first_goals"] if g]
    L += [f"- last: {g}" for g in d["last_goals"] if g]
    L += ["", "## Wellbeing", ""]
    if d["worst_band_ticks"]:
        L += [f"- {k}: {v} ticks in its worst band" for k, v in d["worst_band_ticks"].items()]
    else:
        L.append("- no drive reached its worst band")
    for c in d["guard_pauses"]:
        L.append(f"- **paused by the guard** at {clock(c['t'])}: {c.get('reason')}")
    return "\n".join(L) + "\n"


def write_digest(log: Path, t_from: int, t_to: int) -> Path:
    rows, controls = load(log)
    rows = [r for r in rows if t_from <= r["t"] < t_to]
    controls = [c for c in controls if t_from <= c.get("t", 0) < t_to]
    day = t_from // 86400 + 1
    out = log.with_name(f"{log.stem}.digest-day{day}.md")
    out.write_text(to_markdown(digest_rows(rows, controls), f"{log.stem}: day {day}"), encoding="utf-8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("log")
    ap.add_argument("--from", dest="t_from", type=int, default=0)
    ap.add_argument("--to", dest="t_to", type=int, default=10**12)
    a = ap.parse_args(argv)
    rows, controls = load(Path(a.log))
    rows = [r for r in rows if a.t_from <= r["t"] < a.t_to]
    sys.stdout.write(to_markdown(digest_rows(rows, controls), Path(a.log).stem))
    return 0


if __name__ == "__main__":
    sys.exit(main())
