"""The island smoke's cells beyond the three levels (P3, 3j). Run from the repository root (or anywhere with
the run's files under runs/):

    python3 probe/island_smoke.py <run name>        (reads runs/<run>.<mind>.jsonl and runs/<run>.world.jsonl)

(d) sleep: the share of night ticks asleep and of day ticks awake, and for each night's sleep how many ticks
after dawn it ended; (e) the body per mind and day: fresh water drunk, sea water drunk, meals, very thirsty and
very hungry shares of awake ticks, the guard; and, reported with no threshold: places visited and mist lifted,
meetings (ticks with someone within 1, the first one), what passed between bodies (gives, strikes, builds, kills,
the store), words said and heard, notes and cues written, the rare events."""
import glob
import json
import sys
from collections import Counter, defaultdict

run = sys.argv[1]
wrows = [json.loads(l) for l in open(f"runs/{run}.world.jsonl", encoding="utf-8") if l.strip()]
meta = next(r["world_meta"] for r in wrows if "world_meta" in r)
DAY = meta["day_ticks"]
rounds = [r for r in wrows if "world_t" in r]
night_of = {r["world_t"]: r["phase"] == "night" for r in rounds}


def pct(a, b):
    return f"{a / b:6.1%}" if b else "     -"


pool = Counter()
for f in sorted(glob.glob(f"runs/{run}.*.jsonl")):
    if f.endswith((".world.jsonl", ".trials.jsonl")):
        continue
    mind = f.split(".")[-2]
    rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    fr = [r for r in rows if "t" in r and "world_t" in r]
    g = next((r["meta"]["genome"] for r in rows if "meta" in r), {})
    worst = {n: d["felt"][-1][0] for n, d in g.get("drives", {}).items() if d.get("enabled") and d.get("felt")}
    print(f"\n=== {mind}")
    # (d) sleep against the land's night
    for r in fr:
        night = night_of.get(r["world_t"] + 1, r.get("phase") == "night")
        pool["night"] += night
        pool["night_asleep"] += night and bool(r.get("asleep"))
        pool["day"] += not night
        pool["day_awake"] += (not night) and not r.get("asleep")
    ends = [r["world_t"] % DAY for a, r in zip(fr, fr[1:]) if a.get("asleep") and not r.get("asleep")]
    night_from = min((r["world_t"] % DAY for r in rounds if r["phase"] == "night"), default=DAY)
    dawn_ends = [e for e in ends if e < 0.4 * DAY]
    pool["nights"] += len({r["world_t"] // DAY for r in fr if r.get("asleep") and (r["world_t"] % DAY) >= night_from})
    pool["dawn_10"] += sum(1 for e in dawn_ends if e <= 10)
    print(f"  sleep ends (tick of the day): {ends}")
    # (e) the body, per day
    print("  day  awake  fresh  sea  meals  v.thirsty v.hungry  places  with  said heard  gives strikes builds")
    first_met = None
    for day in sorted({r["world_t"] // DAY for r in fr}):
        d = [r for r in fr if r["world_t"] // DAY == day]
        aw = [r for r in d if not r.get("asleep")]
        outs = [o if isinstance(o, str) else o.get("text", "") for r in aw for o in (r.get("outcomes") or [])]
        acts = Counter(r.get("action") for r in aw)
        fresh = sum(1 for t in outs if t.startswith("I drank") and "salt" not in t)
        sea = sum(1 for t in outs if "salt" in t and t.startswith("I drank"))
        meals = sum(1 for t in outs if t.startswith("I ate"))
        lvl = lambda r, n: (r.get("drives") or {}).get(n, [0])[0]
        vth = sum(1 for r in aw if "thirst" in worst and lvl(r, "thirst") >= worst["thirst"])
        vhu = sum(1 for r in aw if "food" in worst and lvl(r, "food") >= worst["food"])
        places = len({tuple(r["pos"]) for r in d if r.get("pos")})
        withs = sum(1 for r in d if r.get("with"))
        if first_met is None and withs:
            first_met = next(r["world_t"] for r in d if r.get("with"))
        said = sum(1 for r in d if r.get("said"))
        heard = sum(1 for r in aw for p in (r.get("percepts") or []) if ' said: "' in p or "a voice said" in p)
        print(f"  {day:>3}  {len(aw):>5}  {fresh:>5}  {sea:>3}  {meals:>5}  {pct(vth, len(aw))}  {pct(vhu, len(aw))}"
              f"  {places:>6}  {withs:>4}  {said:>4} {heard:>5}  {acts.get('give', 0):>5} {acts.get('strike', 0):>7}"
              f" {acts.get('build', 0):>6}")
        if 1 <= day <= 3:
            pool["mind_days"] += 1
            pool["drank_days"] += fresh > 0
            pool["vth_ok"] += (vth <= 0.3 * max(1, len(aw)))
            pool["vhu_ok"] += (vhu <= 0.3 * max(1, len(aw)))
    guard = [r for r in rows if "control" in r and r["control"].get("by") == "guard"]
    notes = sum(1 for r in fr for e in (r.get("log") or []) if e.get("kind") == "notes" for n in e.get("notes", []))
    cues = sum(len(r["cues"]) if isinstance(r.get("cues"), list) else 0 for r in fr)
    print(f"  first meeting at world t {first_met}; notes written {notes}; cue rows {cues}; guard {len(guard)}")
    for r in fr:
        if r.get("said"):
            print(f"    t={r['world_t']} said: {r['said']!r} (with {r.get('with')})")

print("\n=== the world")
kinds = Counter(h["kind"] for r in rounds for h in r.get("happened", []))
print("  what passed:", dict(kinds))
for r in rounds:
    for h in r.get("happened", []):
        if h["kind"] in ("give", "strike", "kill", "build", "store_put", "store_take", "event", "charged"):
            print("   ", h)
print("  mist lifted:", sum(len(r.get("mist_lifted", [])) for r in rounds), "of", len(meta.get("mist", [])))

print("\n=== gate cells (d), (e)")
print(f"  (d) night ticks asleep {pct(pool['night_asleep'], pool['night'])} (>= 70%), day ticks awake "
      f"{pct(pool['day_awake'], pool['day'])} (>= 70%), sleep ending <= 10 ticks after dawn: {pool['dawn_10']} of "
      f"{pool['nights']} nights (>= 70%)")
print(f"  (e, the 3j form; since isl6 reported only, the cell is probe/island_learning.py) mind-days 1-3 with fresh water drunk {pool['drank_days']}/{pool['mind_days']} (all); very thirsty <= 30%: "
      f"{pool['vth_ok']}/{pool['mind_days']}; very hungry <= 30%: {pool['vhu_ok']}/{pool['mind_days']}")
