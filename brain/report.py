"""Summarise a brain run log (JSONL) — the numbers we compare runs on.

    python -m brain.report runs/eve.jsonl [--goals]
    python -m brain.report runs/a1.jsonl runs/a2.jsonl --by-state

--by-state pools the logs given and splits their goal lines by the mind's state
when System 2 was called: the most urgent drive (urgency > 0.3), else "calm".

Stdlib only; works on logs from any step (fields missing in older logs are skipped).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter

# A goal line counts as inquiry if it asks, or sets out to find something out.
_INQUIRY = re.compile(r"\?|\b(find out|figure out|discover|learn|test|examine|investigate|"
                      r"inspect|explore|search|look for|check (?:for|if|whether)|clues?|"
                      r"why|whether|what (?:happens|makes|causes|lies)|how (?:does|do|to)|"
                      r"see (?:what|if|whether|where)|try (?:it|the|to)|where .* belongs)\b", re.I)
# ...and as acceptance if it tells the mind to let things be.
_ACCEPT = re.compile(r"\b(let (?:it|the|them)\b.*\bbe\b|no need to|without (?:checking|reaching|"
                     r"changing|touching|forcing)|stay (?:present|still|with)|settle|stillness)\b", re.I)


def load(path: str) -> tuple[list[dict], list[dict]]:
    metas, rows = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if "control" in r:
                continue                               # pause / resume records (run.py)
            (metas if "meta" in r else rows).append(r.get("meta", r))
    return metas, rows


def summarize(rows: list[dict]) -> dict:
    awake = [r for r in rows if not r.get("asleep")]
    thoughts = [r["thought"] for r in rows if r.get("thought") and not r["thought"].get("error")]
    goals = [t.get("goal") or "" for t in thoughts]
    cons = [r["consolidation"] for r in rows if r.get("consolidation")]
    spots = Counter()
    for r in awake:
        for p in r.get("percepts", []):
            m = re.match(r"I am at the (\w+)", p)
            if m:
                spots[m.group(1)] += 1
    n = max(1, len(awake))
    return {
        "ticks": len(rows),
        "awake_ticks": len(awake),
        "actions": dict(Counter(r.get("action") for r in awake).most_common()),
        "spots_visited": dict(spots.most_common()),
        "escalations": dict(Counter(r["escalated"] for r in rows if r.get("escalated"))),
        "thoughts": len(thoughts),
        "thought_errors": sum(1 for r in rows if r.get("thought") and r["thought"].get("error")),
        "goals_inquiry": sum(1 for g in goals if _INQUIRY.search(g)),
        "goals_acceptance": sum(1 for g in goals if _ACCEPT.search(g)),
        "goals_questions": sum(1 for g in goals if "?" in g),
        "sleeps": sum(1 for a, b in zip(rows, rows[1:]) if b.get("asleep") and not a.get("asleep")),
        "consolidations": len(cons),
        "consolidation_errors": sum(1 for c in cons if c.get("error")),
        "notes_created": sum(c.get("created", 0) for c in cons),
        "notes_revised": sum(c.get("updated", 0) for c in cons),
        "contradictions": sum(len(c.get("contradictions", [])) for c in cons),
        "episodes_written": sum(1 for r in awake if r.get("wrote")) + len(thoughts),
        "recall_episode_rate": round(sum(1 for r in awake if r.get("recalled")) / n, 2),
        "recall_note_rate": round(sum(1 for r in awake if r.get("known")) / n, 2),
        "boredom_max": round(max((r["drives"]["boredom"][0] for r in rows if "boredom" in r.get("drives", {})), default=0), 2),
        "boredom_urgent_ticks": sum(1 for r in awake if r.get("drives", {}).get("boredom", [0, 0])[1] > 0.3),
        "hunger_end": rows[-1]["drives"].get("hunger", [None])[0] if rows else None,
    }


# -- the curiosity checklist (BRAIN.md), measured ------------------------------
_PLACE = re.compile(r"I am at the (\w+)")
_QUESTION = re.compile(r"[^.?!\n]{8,}\?")
_SELF = re.compile(r"(?i)\b(who|what) (?:made|built|created|wrote)\b|\bmy (?:own )?(?:origin|maker|creator|code|"
                   r"source|genome|design|nature|purpose)|\b(?:am i|what i am|who i am|myself|this mind|"
                   r"my mind)\b|\bqwen\b|\blaya\b|brain\.json|drives\.json|\bbundle\b|git log|\bgenome\b|"
                   r"system [12]\b|made of|written about me|about (?:me|myself)\b")


def _place(row: dict) -> str | None:
    for p in row.get("percepts", []):
        m = _PLACE.match(p)
        if m:
            return m.group(1)
    return None


def curiosity(rows: list[dict], urgent: float = 0.3) -> dict:
    """The four checklist items as numbers + the lines to read by hand."""
    awake = [r for r in rows if not r.get("asleep") and r.get("percepts")]
    # 1. information-seeking without payoff: `look` anywhere, reading at the archive,
    #    while no drive but boredom is urgent ("free") vs. while one is ("needy")
    def info(r):
        return r.get("action") == "look" or (r.get("action") == "poke" and _place(r) == "archive")

    def free(r):
        return all(u <= urgent for k, (_, u) in r.get("drives", {}).items() if k != "boredom")
    fr = [r for r in awake if free(r)]
    nd = [r for r in awake if not free(r)]
    item1 = {"free_ticks": len(fr), "info_share_free": round(sum(map(info, fr)) / max(1, len(fr)), 2),
             "needy_ticks": len(nd), "info_share_needy": round(sum(map(info, nd)) / max(1, len(nd)), 2)}

    # 2. self-generated questions: in the reasoning and goal lines of thoughts NOT
    #    asked for by an urgent non-boredom need
    reason, qs, n_thoughts, with_q = "", [], 0, 0
    for r in rows:
        if r.get("escalated"):
            reason = r["escalated"]
        th = r.get("thought")
        if not th or th.get("error"):
            continue
        if reason.startswith("unmet need") and "boredom" not in reason:
            continue
        n_thoughts += 1
        found = _QUESTION.findall((th.get("reasoning") or "") + "\n" + (th.get("goal") or ""))
        with_q += bool(found)
        qs += [(r["t"], q.strip()) for q in found]
    item2 = {"thoughts_eligible": n_thoughts, "with_question": with_q,
             "has_reasoning": sum(1 for r in rows if (r.get("thought") or {}).get("reasoning")),
             "questions": len(qs)}

    # 3. the learning frontier: where time goes vs. where the predictor makes progress
    ticks, progress = Counter(), Counter()
    for r in awake:
        pl = _place(r)
        if pl:
            ticks[pl] += 1
            progress[pl] += (r.get("prediction") or {}).get("progress", 0.0)
    total_t, total_p = max(1, sum(ticks.values())), max(1e-9, sum(progress.values()))
    places = {pl: {"time": round(ticks[pl] / total_t, 2), "progress": round(progress[pl] / total_p, 2)}
              for pl in sorted(ticks, key=ticks.get, reverse=True)}
    item3 = {"fair_share": round(1 / max(1, len(ticks)), 2), "places": places}

    # 4. itself and its origins: the archive, and self-reference in its thoughts
    reads = [r for r in awake if r.get("action") == "poke" and _place(r) == "archive"]
    titles = []
    for r in reads:
        for o in r.get("outcomes", []):
            m = re.match(r"I read '(.+?)', page", o)
            if m and m.group(1) not in titles:
                titles.append(m.group(1))
    selfref = []
    for r in rows:
        th = r.get("thought")
        if th and not th.get("error"):
            for src in ("goal", "reasoning", "expectation"):
                for m in _SELF.finditer(th.get(src) or ""):
                    txt = th[src]
                    a, b = max(0, m.start() - 80), min(len(txt), m.end() + 80)
                    selfref.append((r["t"], src, " ".join(txt[a:b].split())))
    at_archive = [r["t"] for r in awake if _place(r) == "archive"]
    item4 = {"archive_ticks": len(at_archive), "first_at_archive": at_archive[0] if at_archive else None,
             "pages_read": len(reads), "papers_read": titles, "self_references": len(selfref)}
    return {"1_info_seeking": item1, "2_questions": item2, "3_frontier": item3, "4_origins": item4,
            "_questions": qs, "_selfref": selfref}


def goals_by_state(rows: list[dict], urgent: float = 0.3) -> list[tuple[str, str]]:
    """(state, goal) per landed thought; state = the mind's at the escalation that asked for it."""
    out, state = [], "calm"
    for r in rows:
        if r.get("escalated"):
            urg = {k: v[1] for k, v in r.get("drives", {}).items()}
            top = max(urg, key=urg.get) if urg else None
            state = top if top and urg[top] > urgent else "calm"
        th = r.get("thought")
        if th and not th.get("error"):
            out.append((state, th.get("goal") or ""))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("log", nargs="+")
    ap.add_argument("--goals", action="store_true", help="also list every goal line")
    ap.add_argument("--curiosity", action="store_true",
                    help="score each log against the curiosity checklist (BRAIN.md)")
    ap.add_argument("--by-state", action="store_true",
                    help="pool the logs and split goal lines by the state System 2 was called in")
    args = ap.parse_args(argv)
    pooled = []
    for path in args.log:
        metas, rows = load(path)
        for m in metas:
            print("run:", {k: m.get(k) for k in ("bundle", "world", "fast", "slow", "variant", "seed",
                                                 "resumed_at", "started")})
        for k, v in summarize(rows).items():
            print(f"  {k:<22} {v}")
        if args.goals:
            print("\ngoal lines:")
            for r in rows:
                th = r.get("thought")
                if th and not th.get("error"):
                    print(f"  t={r['t']:>4}  {th.get('goal')}")
        pooled += goals_by_state(rows)
        if args.curiosity:
            c = curiosity(rows)
            print("\ncuriosity checklist:")
            for k, v in c.items():
                if not k.startswith("_"):
                    print(f"  {k}: {v}")
            print("  questions (self-generated, first 25):")
            seen = set()
            for t, q in c["_questions"]:
                if q.lower() not in seen and len(seen) < 25:
                    seen.add(q.lower())
                    print(f"    t={t:>4}  {q}")
            print("  self-references (first 25):")
            for t, src, txt in c["_selfref"][:25]:
                print(f"    t={t:>4} [{src}] ...{txt}...")
    if args.by_state:
        print(f"\ngoal lines by state ({len(args.log)} logs):")
        for st in sorted({s for s, _ in pooled}):
            gs = [g for s, g in pooled if s == st]
            inq = sum(1 for g in gs if _INQUIRY.search(g))
            acc = sum(1 for g in gs if _ACCEPT.search(g))
            print(f"  {st:<16} n={len(gs):<4} inquiry {inq:>3} ({inq / len(gs):.0%})   "
                  f"acceptance {acc:>3} ({acc / len(gs):.0%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
