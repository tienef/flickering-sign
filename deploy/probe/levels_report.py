"""Three levels, read from a run's logs (P1 step 7 smoke and after). Run from the repository root:

    python3 probe/levels_report.py <run name>            (reads runs/<run>.<mind>.jsonl)

Per mind and day: awake ticks, the fast level's flag rate and who chose (slow / middle / head / carry, and how it
carried), the signals that fired, the middle level's latency, errors and letter mass, the slow level's thoughts
(asked, landed, cut, seconds, reply without JSON), episodes written, the head's agreement and nights, welfare
(the share of awake ticks in each drive's worst felt band, very hungry, guard). And the gate at the moments where
the act matters (knowledge_act's door, bush, food lying, food held): the share of moment runs whose first tick
was flagged."""
import glob
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from knowledge_act import moments  # noqa: E402

run = sys.argv[1]


def pct(a, b):
    return f"{a / b:6.1%}" if b else "    -"


# a shared land's run (together.py, P3) has no trial days: a day is the land's day, from the world's log
wf = Path(f"runs/{run}.world.jsonl")
day_ticks = (next((json.loads(l)["world_meta"]["day_ticks"] for l in open(wf) if '"world_meta"' in l), None)
             if wf.exists() else None)

for f in sorted(glob.glob(f"runs/{run}.*.jsonl")):
    if f.endswith(".trials.jsonl") or f.endswith(".world.jsonl"):
        continue
    mind = f.split(".")[-2]
    rows = [json.loads(l) for l in open(f) if l.strip()]
    meta = next((r["meta"] for r in rows if "meta" in r), {})
    g = meta.get("genome") or {}
    worst = {n: d["felt"][-1][0] for n, d in g.get("drives", {}).items() if d.get("enabled") and d.get("felt")}
    frames = [r for r in rows if "t" in r]
    print(f"\n=== {mind}  levels {g.get('levels')}  head {g.get('head')}")
    by_day = defaultdict(list)
    for r in frames:
        if day_ticks and "day" not in r and "world_t" in r:
            r["day"] = r["world_t"] // day_ticks
        by_day[r.get("day")].append(r)
    tot = Counter()
    secs, mass, tsecs, why_all = [], [], [], Counter()
    runs_n = runs_flag = 0
    prev_m = None
    for day, fr in sorted(by_day.items(), key=lambda kv: (kv[0] is None, kv[0])):
        aw = [r for r in fr if not r.get("asleep") and r.get("fast")]
        n = len(aw)
        by = Counter(r["fast"]["by"] for r in aw)
        carry = Counter(r["fast"].get("carry") for r in aw if r["fast"]["by"] == "carry")
        flags = sum(r["fast"]["flag"] for r in aw)
        errs = sum(1 for r in aw if r["fast"].get("error"))
        heads = [r["fast"]["head"] for r in aw if "agreed" in (r["fast"].get("head") or {})]
        for r in aw:
            why_all.update(r["fast"]["why"].keys())
            if "seconds" in r["fast"]:
                secs.append(r["fast"]["seconds"])
            if "mass" in r["fast"]:
                mass.append(r["fast"]["mass"])
        th = [r["thought"] for r in fr if r.get("thought") and r["thought"].get("kind") == "thought"]
        tsecs += [t.get("seconds", 0) for t in th]
        nojson = sum(1 for t in th if t.get("error"))
        asked = sum(1 for r in fr for e in r.get("log", []) if e["kind"] == "thought_asked")
        cut = sum(1 for r in fr for e in r.get("log", []) if e["kind"] == "thought_cut")
        eps = sum(1 for r in fr for e in r.get("log", []) if e["kind"] == "episode")
        bad = {d: sum(1 for r in aw if (r.get("drives") or {}).get(d, [0])[0] >= th_) for d, th_ in worst.items()}
        hungry = sum(1 for r in aw if (r.get("drives") or {}).get("food", [0])[0] >= 0.8)
        print(f"  day {day}: awake {n:4d}  flag {pct(flags, n)}  by " + " ".join(f"{k} {pct(v, n)}" for k, v in by.most_common())
              + f"  carry {dict(carry)}  middle err {errs}  thoughts asked {asked} landed {len(th)} cut {cut}"
              f" no-json {nojson}  episodes {eps}  head agreed {pct(sum(h['agreed'] for h in heads), len(heads))}"
              f" ({len(heads)})  very hungry {pct(hungry, n)}  worst " + " ".join(f"{d} {pct(v, n)}" for d, v in bad.items() if v))
        for r in fr:
            if r.get("head_night"):
                hn = r["head_night"]
                print(f"    head night: labels {hn.get('labels')} teach {hn.get('teach')} held-out agree "
                      f"{hn.get('agree_before')} -> {hn.get('agree_after')} accepted {hn.get('accepted')} "
                      f"{hn.get('why', '')} {hn.get('seconds')} s")
            if r.get("distress_pause"):
                print(f"    GUARD PAUSE: {r['distress_pause']}")
        for r in aw:                                   # moment runs: the first tick of each
            m = tuple(moments(r))
            if m and m != prev_m:
                runs_n += 1
                runs_flag += bool(r["fast"]["flag"])
            prev_m = m
        tot.update(by)
        tot["awake"] += n
        tot["flag"] += flags
    q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))] if xs else None
    print(f"  all: flag {pct(tot['flag'], tot['awake'])}; middle latency p50 {q(secs, .5)} p90 {q(secs, .9)} s, "
          f"letter mass p10 {q(mass, .1)}; thought seconds p50 {q(tsecs, .5)} p90 {q(tsecs, .9)}; "
          f"moment onsets flagged {runs_flag}/{runs_n} {pct(runs_flag, runs_n)}")
    print(f"  signals fired (ticks): {dict(why_all.most_common())}")
