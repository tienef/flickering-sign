"""The island smoke's cell (e), rewritten to measure learning over the days (owner, 2026-09-30, E). Run from the
repository root:

    python3 probe/island_learning.py <run name> [split day]     (runs/<run>.<mind>.jsonl; split default 3)

Per mind: (welfare) very hungry and very thirsty shares of awake ticks over days 1..end; (e1) the day of the first
meal and of the first fresh drink; (e2) each time a need is first felt (food or thirst crossing its first felt band),
the awake ticks until it is relieved (a meal; a drink of fresh water), unrelieved ones counted to the run's end,
median before the split day and from it; (e3) sea water drunk before and from the split day; (e4) the notes written
about food and water, quoted, to be read (true or not is judged by us, not scored)."""
import glob
import json
import re
import sys
from statistics import median

run = sys.argv[1]
SPLIT = int(sys.argv[2]) if len(sys.argv) > 2 else 3
wrows = [json.loads(l) for l in open(f"runs/{run}.world.jsonl", encoding="utf-8") if l.strip()]
DAY = next(r["world_meta"] for r in wrows if "world_meta" in r)["day_ticks"]
WORDS = re.compile(r"eat|food|hung|mussel|berr|apple|fish|meat|drink|drank|water|spring|thirst|sea|salt", re.I)


def txt(o):
    return o if isinstance(o, str) else o.get("text", "")


def fresh(t):
    return t.startswith("I drank") and "salt" not in t


pool = {"minds": 0, "welfare": 0, "e1": 0, "e2_food": 0, "e2_thirst": 0, "sea_early": 0, "sea_late": 0}
for f in sorted(glob.glob(f"runs/{run}.*.jsonl")):
    if f.endswith((".world.jsonl", ".trials.jsonl")):
        continue
    mind = f.split(".")[-2]
    rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    fr = [r for r in rows if "t" in r and "world_t" in r]
    g = next((r["meta"]["genome"] for r in rows if "meta" in r), {})
    band = {n: (d["felt"][0][0], d["felt"][-1][0]) for n, d in g.get("drives", {}).items()
            if n in ("food", "thirst") and d.get("enabled") and d.get("felt")}
    lvl = lambda r, n: (r.get("drives") or {}).get(n, [0])[0]
    day = lambda r: r["world_t"] // DAY
    pool["minds"] += 1
    print(f"\n=== {mind}  (felt, worst) {band}")
    # welfare, days 1..end
    aw = [r for r in fr if not r.get("asleep") and day(r) >= 1]
    share = {n: sum(1 for r in aw if lvl(r, n) >= band[n][1]) / max(1, len(aw)) for n in band}
    ok = all(v <= 0.2 for v in share.values())
    pool["welfare"] += ok
    print("  welfare days 1+: " + ", ".join(f"very {n} {v:.1%}" for n, v in share.items()) + f"  (<= 20%: {ok})")
    # e1: first meal, first fresh drink
    first = {"meal": None, "fresh": None}
    for r in fr:
        for o in r.get("outcomes") or []:
            t = txt(o)
            if first["meal"] is None and t.startswith("I ate"):
                first["meal"] = day(r)
            if first["fresh"] is None and fresh(t):
                first["fresh"] = day(r)
    e1 = all(v is not None and v <= 2 for v in first.values())
    pool["e1"] += e1
    print(f"  first meal on day {first['meal']}, first fresh drink on day {first['fresh']}  (both by day 2: {e1})")
    # e2: need felt -> relieved, in awake ticks
    for n, relief in (("food", lambda t: t.startswith("I ate")), ("thirst", fresh)):
        if n not in band:
            continue
        eps, onset, awake = [], None, 0
        prev = None
        for r in fr:
            lv = lvl(r, n)
            if onset is None and prev is not None and prev < band[n][0] <= lv:
                onset, awake = day(r), 0
            if onset is not None:
                awake += not r.get("asleep")
                if any(relief(txt(o)) for o in r.get("outcomes") or []):
                    eps.append((onset, awake, True))
                    onset = None
            prev = lv
        if onset is not None:
            eps.append((onset, awake, False))
        early = [a for d, a, _ in eps if d < SPLIT]
        late = [a for d, a, _ in eps if d >= SPLIT]
        better = bool(early and late and median(late) < median(early))
        pool[f"e2_{n}"] += better
        show = " ".join(f"d{d}:{a}{'' if ok_ else '+'}" for d, a, ok_ in eps)
        print(f"  {n}: felt -> relieved (awake ticks; + = not relieved by the end): {show or '-'}"
              f"  median before day {SPLIT} {median(early) if early else '-'}, from it {median(late) if late else '-'}"
              f"  (faster: {better})")
    # e3: sea water
    sea = [day(r) for r in fr for o in r.get("outcomes") or [] if txt(o).startswith("I drank") and "salt" in txt(o)]
    se, sl = sum(d < SPLIT for d in sea), sum(d >= SPLIT for d in sea)
    pool["sea_early"] += se
    pool["sea_late"] += sl
    print(f"  sea water drunk: before day {SPLIT} {se}, from it {sl}")
    # e4: notes about food and water, to read
    seen = set()
    for r in fr:
        for e in r.get("log") or []:
            if e.get("kind") != "notes":
                continue
            for nt in e.get("notes", []):
                body = nt.get("text") or nt.get("body") or json.dumps(nt, ensure_ascii=False)
                title = nt.get("title", "")
                if (title, body[:80]) in seen or not WORDS.search(title + " " + body):
                    continue
                seen.add((title, body[:80]))
                print(f"    t={r['world_t']} [{title}] {body[:220]}")

m = pool["minds"]
print(f"\n=== gate cell (e), learning (split day {SPLIT})")
print(f"  welfare: very hungry and very thirsty <= 20% of awake ticks from day 1: {pool['welfare']}/{m} minds (all)")
print(f"  (e1) ate and drank fresh water by day 2: {pool['e1']}/{m} (all)")
print(f"  (e2) need relieved faster from day {SPLIT}: food {pool['e2_food']}/{m}, thirst {pool['e2_thirst']}/{m} (>= 3 each)")
print(f"  (e3) sea water drunk: before day {SPLIT} {pool['sea_early']}, from it {pool['sea_late']} (fewer from it)")
print("  (e4) notes on food and water: read above (quoted in PLAN)")
