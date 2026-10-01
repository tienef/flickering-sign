"""Habits (step 5c): sleep retunes System 1.

Laya is a cortex and a striatum: its encoder (ModernBERT-large, 395 M parameters) is general,
shared and stays frozen; its decision head (two transformer layers + the option scorer, ~25 M)
is where a choice is made, and each mind gets its own copy, tuned by what followed its choices.
While awake, every appraisal is kept in the day's replay buffer; falling asleep replays it and
the head learns (Complementary Learning Systems: the hippocampus replays the day, the slow
system learns from it; sharp-wave replay). Two teachers, weighted (owner's choice 1c):

- **distillation**: where System 2's thought chose the action, that action's value in that state
  is pushed toward "very well" (the prefrontal cortex teaches the striatum: behaviour starts
  goal-directed and becomes habit with repetition, Balleine & Dickinson 1998);
- **reinforcement**: a value target from what followed, the discounted return of internal
  rewards (`habits.reward`: "liking" -- eating, a task done, an expectation met, learning
  progress -- and effectance, owner's choice), relative to the day's mean (dopamine's reward
  prediction error, Schultz 1997), so the chain that led to a good moment gets credit backwards.

Step 10x (after s10: the day-mean baseline taught the ripple of learning progress, not successes):
`baseline: 0` makes the reward sparse -- only the ticks within the credit window BEFORE a reward
teach (their value rises with the return), an ordinary tick teaches nothing, as phasic dopamine
answers unexpected rewards and not the steady trickle of learning; `distill_intention` also
distils the ticks where System 1 carried out System 2's intended verb and it still changed
something (the prefrontal cortex teaches the striatum by repetition, not only once); `keep_days`
keeps each day's replay, with the tick's raw signals (`sig`), so a reward can be re-tuned offline;
`train: false` records and keeps the days without ever training (a control arm that can be replayed).

Each example weighs 1 + its emotional tag (adrenaline, McGaugh 2004). Not forgetting (choice 3):
a sample of older nights' examples is interleaved, the answers to everything else are anchored
to the original head on sampled states, a fixed probe set (drawn the first night, never trained
on) is checked after each night, the steps per night are capped, and a night that drifts too far
from the original on the probe is rolled back (the head is only written when accepted).

Genome: brain.json `habits` (absent = System 1 never changes). Files, in the bundle's `habits/`:
  replay.jsonl      today's appraisals (cleared each night) and adrenaline's tags
  days/             each past day's replay, gzipped (keep_days), for offline tuning
  old.jsonl         a sample of older nights' training examples (interleaving)
  probe.jsonl       the fixed probe set
  head.safetensors  this mind's decision head (absent until its first accepted night)
  nights.jsonl      what each night did
"""
from __future__ import annotations

import gzip
import json
import math
import random
import time
from pathlib import Path

HEAD_PARTS = ("head", "type_emb", "scorer")     # the striatum; the encoder and act head stay shared
LEVELS = 3                                        # "not at all / somewhat / very well" (and salience's three)

DEFAULTS = {
    "reward": {"ate.amount": 2.0, "task_done": 0.6, "expectation.met": 0.3,
               "prediction.progress": 0.5, "effectance.amount": 1.0},
    "gamma": 0.93,             # credit halves every ~10 ticks back
    "value_step": 0.3,         # a return one standard deviation above the day's mean moves the target 0.3 (0..2)
    "z_min": 0.5,              # rows closer to the mean teach nothing (the anchor holds them)
    "z_max": 2.0,
    "baseline": "day_mean",    # or 0: sparse, only the ticks before a reward teach (step 10x)
    "min_return": 0.05,        # baseline 0: a return below this teaches nothing
    "distill_intention": False,  # also distil the ticks System 1 followed the intention with effect
    "keep_days": True,         # gzip each day's replay into habits/days/ before clearing it
    "train": True,             # False: record and keep the days, never train (a control arm)
    "distill_score": 1.8,      # System 2's choice: toward "very well"
    "max_reinforce": 200,
    "max_distill": 100,
    "old_keep": 400,           # older nights' examples kept for interleaving
    "old_per_night": 100,
    "anchor_rows": 64,         # states whose other answers are held to the original head
    "anchor_verbs": 3,         # value questions per anchor state (plus salience)
    "anchor_weight": 1.0,
    "probe_rows": 24,
    "max_drift": 0.3,          # mean |score change| vs the original head on the probe (0..2 scale)
    "lr": 3e-4,                # calibrated on the first real night (deploy/probe/habits_tune.py): a taught
    "epochs": 4,               # value rises ~0.075 a night (x2 its odds at choice temperature 0.12), probe drift ~0.04
    "batch": 16,
    "max_steps": 200,
}


def settings(cfg: dict | None) -> dict:
    out = dict(DEFAULTS)
    for k, v in (cfg or {}).items():
        if not k.startswith("_"):
            out[k] = v
    return out


def reward(cfg: dict, summary: dict) -> float:
    """The tick's internal reward: a weighted sum of its event signals (as a dial's gains)."""
    return round(sum(float(g) * summary.get(sig, 0.0) for sig, g in cfg["reward"].items()
                     if not sig.startswith("_")), 5)


def target_dist(score: float) -> list[float]:
    """A distribution over the three levels whose expected level is `score` (0..2)."""
    s = min(max(score, 0.0), LEVELS - 1.0)
    lo = min(int(math.floor(s)), LEVELS - 2)
    p = [0.0] * LEVELS
    p[lo], p[lo + 1] = 1.0 - (s - lo), s - lo
    return [round(x, 4) for x in p]


def returns(rows: list[dict], gamma: float) -> list[float]:
    out, g = [0.0] * len(rows), 0.0
    for i in range(len(rows) - 1, -1, -1):
        g = float(rows[i].get("r", 0.0)) + gamma * g
        out[i] = g
    return out


class Store:
    """The bundle's `habits/` folder."""

    def __init__(self, path: Path):
        self.dir = Path(path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.replay = self.dir / "replay.jsonl"
        self.head = self.dir / "head.safetensors"

    def _append(self, name: str, obj: dict) -> None:
        with open(self.dir / name, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _read(self, name: str) -> list[dict]:
        p = self.dir / name
        if not p.exists():
            return []
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass                                      # a line cut by a power-off
        return out

    def _write(self, name: str, objs: list[dict]) -> None:
        tmp = self.dir / (name + ".tmp")
        tmp.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in objs), encoding="utf-8")
        tmp.replace(self.dir / name)

    # the day
    def add(self, row: dict) -> None:
        self._append("replay.jsonl", row)

    def tag(self, since: int, upto: int, tag: float) -> None:
        self._append("replay.jsonl", {"tag": tag, "since": since, "upto": upto})

    def day(self) -> list[dict]:
        rows, tags = [], []
        for o in self._read("replay.jsonl"):
            (tags if "tag" in o else rows).append(o)
        for tg in tags:                                   # the highest surge counts, as for episodes
            for r in rows:
                if tg["since"] <= r["t"] <= tg["upto"] and tg["tag"] > r.get("tag", 0.0):
                    r["tag"] = tg["tag"]
        return sorted(rows, key=lambda r: r["t"])

    def clear_day(self) -> None:
        self._write("replay.jsonl", [])

    def archive_day(self, name: str) -> None:
        """The day's replay as written (rows and tags), gzipped, before it is cleared."""
        if not self.replay.exists() or not self.replay.stat().st_size:
            return
        days = self.dir / "days"
        days.mkdir(exist_ok=True)
        with open(self.replay, "rb") as src, gzip.open(days / f"{name}.jsonl.gz", "wb") as dst:
            dst.write(src.read())

    # the nights
    def old(self) -> list[dict]:
        return self._read("old.jsonl")

    def keep_old(self, examples: list[dict], keep: int, rng: random.Random) -> None:
        pool = self.old() + examples
        if len(pool) > keep:
            pool = rng.sample(pool, keep)
        self._write("old.jsonl", pool)

    def probe(self) -> list[dict]:
        return self._read("probe.jsonl")

    def set_probe(self, rows: list[dict]) -> None:
        self._write("probe.jsonl", [{"t": r["t"], "state": r["state"], "verbs": r["verbs"]} for r in rows])

    def night(self, rep: dict) -> None:
        self._append("nights.jsonl", rep)


def build_examples(rows: list[dict], cfg: dict, rng: random.Random) -> tuple[list[dict], dict]:
    """The night's teaching examples from the day's rows: reinforcement (a value target from the
    return, relative to the day's mean) and distillation (System 2's choices)."""
    rows = [r for r in rows if r.get("raw")]
    stats = {"rows": len(rows)}
    if not rows:
        return [], stats
    G = returns(rows, float(cfg["gamma"]))
    mean = sum(G) / len(G)
    std = math.sqrt(sum((g - mean) ** 2 for g in G) / len(G))
    rewarded = sum(1 for r in rows if r.get("r", 0.0) > 0)
    stats.update(return_mean=round(mean, 4), return_std=round(std, 4), rewarded_ticks=rewarded,
                 reward_sum=round(sum(r.get("r", 0.0) for r in rows), 4))

    reinforce = []
    zmax = float(cfg["z_max"])
    sparse = cfg.get("baseline") == 0
    for r, g in zip(rows, G):
        a = r.get("action")
        if a not in r["raw"] or r.get("by") not in ("s1", "s2"):
            continue                                      # a forced act (collapse, an urge held back) was not chosen
        if sparse:
            if g < float(cfg["min_return"]):
                continue                                  # no reward ahead: nothing to learn here
            z = min(zmax, g)                              # the return itself, in reward units
        elif std > 1e-6:
            z = max(-zmax, min(zmax, (g - mean) / std))
            if abs(z) < float(cfg["z_min"]):
                continue
        else:
            continue
        score = r["raw"][a] + float(cfg["value_step"]) * z
        reinforce.append({"kind": "reinforce", "state": r["state"], "verb": a, "desc": r["verbs"][a],
                          "old": r["raw"][a], "target": target_dist(score), "z": round(z, 3),
                          "w": 1.0 + float(r.get("tag", 0.0)), "t": r["t"]})
    reinforce.sort(key=lambda e: -abs(e["z"]))
    reinforce = reinforce[: int(cfg["max_reinforce"])]
    reinforce += variations(rows, reinforce, cfg, rng)

    distill = []
    for r in rows:
        a = r.get("action")
        taught = r.get("by") == "s2" or (cfg.get("distill_intention") and r.get("follows"))
        if taught and a in r["raw"]:
            score = max(r["raw"][a], float(cfg["distill_score"]))
            distill.append({"kind": "distill", "state": r["state"], "verb": a, "desc": r["verbs"][a],
                            "old": r["raw"][a], "target": target_dist(score),
                            "w": 1.0 + float(r.get("tag", 0.0)), "t": r["t"],
                            **({"follows": True} if r.get("by") != "s2" else {})})
    cap = int(cfg["max_distill"])
    if len(distill) > cap:                                # System 2's own choices first, then the intention followed
        own = [e for e in distill if not e.get("follows")]
        rest = [e for e in distill if e.get("follows")]
        own = rng.sample(own, min(cap, len(own)))
        distill = own + rng.sample(rest, min(cap - len(own), len(rest)))
    stats.update(reinforce=len(reinforce), reinforce_up=sum(1 for e in reinforce if e["z"] > 0),
                 varied=sum(1 for e in reinforce if e["kind"] == "vary"),
                 distill=len(distill), distill_follows=sum(1 for e in distill if e.get("follows")),
                 tagged=sum(1 for e in reinforce + distill if e["w"] > 1.0))
    return reinforce + distill, stats


def variations(rows: list[dict], taught: list[dict], cfg: dict, rng: random.Random) -> list[dict]:
    """Replay with variations (step 10z): a rewarded act is also taught on the day's other states
    where the same thing lay before the mind (`variations.match`, a regex on the state's `field`;
    its first group is the thing), as hippocampal replay recombines rather than repeats (Gupta et
    al. 2010; Liu et al. 2019) and a response generalises to like stimuli. Only the day's own
    states: nothing is invented. Acts in `ignore` (moves, looking, waiting) do not depend on what
    lies in front, so they are not varied. `match_by_verb` gives an act its own object (eating is
    about what is held, not what is in front: 10aa, the bench taught "eat facing the wall")."""
    v = cfg.get("variations")
    if not v or not taught:
        return []
    import re
    field = v.get("field", "situation")
    pats = {k: re.compile(x) for k, x in v.get("match_by_verb", {}).items()}
    default = re.compile(v["match"])
    ignore = set(v.get("ignore", ()))

    def thing(r, verb):
        m = pats.get(verb, default).search(str(r["state"].get(field, "")))
        return m.group(1) if m else None

    out = []
    for e in taught:
        if e["z"] <= 0 or e["verb"] in ignore:
            continue
        src = next((r for r in rows if r["t"] == e["t"]), None)
        key = thing(src, e["verb"]) if src else None
        if key is None:
            continue
        like = [r for r in rows if r["t"] != e["t"] and e["verb"] in r["raw"] and thing(r, e["verb"]) == key]
        for r in rng.sample(like, min(int(v.get("per_example", 8)), len(like))):
            out.append({"kind": "vary", "state": r["state"], "verb": e["verb"], "desc": r["verbs"][e["verb"]],
                        "old": r["raw"][e["verb"]],
                        "target": target_dist(r["raw"][e["verb"]] + float(cfg["value_step"]) * e["z"]),
                        "z": e["z"], "w": float(v.get("weight", 0.5)) * e["w"], "t": r["t"], "from": e["t"]})
    return out


def _expected(p) -> float:
    return float(sum(i * x for i, x in enumerate(p)))


def night(fast, mind: str, store: Store, cfg: dict, seed: str) -> dict:
    """One night's learning for one mind (`fast` is a LayaFast): build the examples, train a copy
    of the mind's head, check the probe, keep or roll back. Clears the day either way."""
    t0 = time.monotonic()
    rng = random.Random(seed)
    day = store.day()
    examples, rep = build_examples(day, cfg, rng)
    rep["night_of"] = day[-1]["t"] if day else None
    if cfg.get("keep_days") and day:
        store.archive_day(f"day-{day[-1]['t']:07d}")
    if not cfg.get("train", True):                      # a recording arm: the day is kept, nothing learns
        rep.update(accepted=False, why="recording only", seconds=round(time.monotonic() - t0, 2))
        store.clear_day()
        store.night(rep)
        return rep
    probe = store.probe()
    if not probe and day:                                 # the first night: draw the fixed probe set
        probe_rows = rng.sample(day, min(int(cfg["probe_rows"]), len(day) // 4))
        store.set_probe(probe_rows)
        held = {r["t"] for r in probe_rows}
        examples = [e for e in examples if e["t"] not in held]
        probe = store.probe()
    old = store.old()
    interleaved = rng.sample(old, min(int(cfg["old_per_night"]), len(old)))
    anchors = rng.sample(day, min(int(cfg["anchor_rows"]), len(day))) if day else []
    rep.update(interleaved=len(interleaved), anchors=len(anchors), probe=len(probe))
    if not examples:
        rep.update(accepted=False, why="nothing to learn", seconds=round(time.monotonic() - t0, 2))
    else:
        rep.update(fast.train_head(mind, examples + interleaved, anchors, probe, cfg, rng, store.head))
        rep["seconds"] = round(time.monotonic() - t0, 2)
    # what was taught tonight joins the older nights' sample (targets kept: interleaving replays them)
    store.keep_old([{k: e[k] for k in ("kind", "state", "verb", "desc", "old", "target", "w", "t", "z") if k in e}
                    for e in examples], int(cfg["old_keep"]), rng)
    store.clear_day()
    store.night(rep)
    return rep


# -- the fast head: distillation of the middle level (three levels, PLAN 1.1 and P1 step 3) ---------------
HEAD_DEFAULTS = {
    "mode": "observe",         # observe: the head proposes, the middle level decides; decide: see below
    "decide_above": 0.9,       # decide: the head acts alone when its choice's probability is at least this ...
    "trust_above": 0.8,        # ... and it agreed with the middle level's top on this share of ...
    "trust_window": 50,        # ... the last flagged ticks where both answered
    "audit": 0.1,              # decide: this share of the ticks it could decide is still asked (its record goes on)
    "held_share": 0.2,         # of each day's labels, kept out of training to judge the nights
    "probe_keep": 200,         # held-out labels kept (the most recent)
    "old_keep": 600,           # older nights' labels kept for interleaving
    "old_per_night": 150,
    "max_worse": 0.02,         # a night that lowers agreement on the held-out labels by more is rolled back
    "lr": 1e-3,                # chosen on 381 states labelled by the FP8 (P1 step 3): after two nights of 150 labels,
    "epochs": 12,              # agreement on unseen states 0.17 -> 0.36 (majority 0.27); 3e-4 x 4 epochs: 0.21
    "batch": 16,
    "max_steps": 1500,
    "keep_days": True,
}


def head_settings(cfg: dict | None) -> dict:
    out = dict(HEAD_DEFAULTS)
    out.update({k: v for k, v in (cfg or {}).items() if not k.startswith("_")})
    return out


def distill_night(fast, mind: str, store: Store, cfg: dict, seed: str) -> dict:
    """One night for the fast head (`fast`: a LayaFast in choice mode): the day's labels are the middle
    level's choices on the flagged ticks (its whole distribution over the acts, not only the act drawn), as
    the prefrontal cortex's choices are rehearsed in sleep until the striatum makes them alone (automatisation;
    ACT-R's proceduralisation, Logan's instances). A share of the day is held out; the night trains on the
    rest with older nights' labels interleaved, and is kept only if the head does not agree worse with the
    middle level on every held-out label kept so far. Clears the day either way."""
    t0 = time.monotonic()
    rng = random.Random(seed)
    day = [r for r in store.day() if r.get("probs") and r.get("verbs")]
    rep = {"night_of": day[-1]["t"] if day else None, "labels": len(day)}
    if day:
        seen = [r for r in day if r.get("head")]
        if seen:                                           # how the head did today, proposing only
            rep["agreed_today"] = round(sum(r["head"] == r["top"] for r in seen) / len(seen), 3)
    if cfg.get("keep_days") and day:
        store.archive_day(f"day-{day[-1]['t']:07d}")
    rng.shuffle(day)
    n_held = int(round(len(day) * float(cfg["held_share"])))
    held, teach = day[:n_held], day[n_held:]
    probe = (store.probe() + [_label(r) for r in held])[-int(cfg["probe_keep"]):]
    store._write("probe.jsonl", probe)
    examples = [_label(r) for r in teach]
    old = store.old()
    interleaved = rng.sample(old, min(int(cfg["old_per_night"]), len(old)))
    rep.update(teach=len(examples), interleaved=len(interleaved), probe=len(probe))
    if not examples:
        rep.update(accepted=False, why="nothing to learn")
    elif not hasattr(fast, "distill_head"):
        rep.update(accepted=False, why="the fast level cannot learn")
    else:
        rep.update(fast.distill_head(mind, examples + interleaved, probe, cfg, rng, store.head))
    rep["seconds"] = round(time.monotonic() - t0, 2)
    store.keep_old(examples, int(cfg["old_keep"]), rng)
    store.clear_day()
    store.night(rep)
    return rep


def _label(r: dict) -> dict:
    """A training label: the state, the acts offered, the middle level's probabilities in the acts' order."""
    verbs = r["verbs"]
    return {"t": r["t"], "state": r["state"], "verbs": verbs,
            "target": [float(r["probs"].get(v, 0.0)) for v in verbs], "top": r.get("top"),
            "w": 1.0 + float(r.get("tag", 0.0))}


# -- the torch side (on the box) -------------------------------------------------------
def encode(agent, items: list[tuple[dict, dict]], batch: int = 16, cache: dict | None = None) -> list[dict]:
    """The frozen encoder's output for each (state, question), computed once (no gradient).
    `cache` (offline replays only): a dict keyed by the item, filled and reused."""
    if cache is not None:
        keys = [json.dumps([s, q], sort_keys=True) for s, q in items]
        miss = {k: it for k, it in zip(keys, items) if k not in cache}
        if miss:
            cache.update(zip(miss, encode(agent, list(miss.values()), batch)))
        return [cache[k] for k in keys]
    import torch
    from laya.common import QTYPES, build_sequence

    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)
    seqs = []
    for state, qdef in items:
        q = agent._to_internal(qdef)
        ids, markers = build_sequence(agent.tok, state, q, max_len, head_max_len)
        seqs.append((ids, markers, QTYPES[q["t"]]))
    out = []
    model, dev = agent.model, agent.device
    for i in range(0, len(seqs), batch):
        chunk = seqs[i:i + batch]
        L = max(len(s[0]) for s in chunk)
        ids = torch.full((len(chunk), L), agent.tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for j, (s, _, _) in enumerate(chunk):
            ids[j, :len(s)] = torch.tensor(s)
            att[j, :len(s)] = 1
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=agent.dtype, enabled=dev.type == "cuda"):
            h = model.encoder(input_ids=ids.to(dev), attention_mask=att.to(dev)).last_hidden_state
        for j, (s, markers, qt) in enumerate(chunk):
            out.append({"h": h[j, :len(s)].to(torch.float16), "markers": markers, "qtype": qt})
    return out


def head_logits(parts: dict, enc: list[dict], temperature: float):
    """The head's option logits (divided by Laya's temperature) for a batch of encoded items."""
    import torch

    dev = enc[0]["h"].device
    L = max(e["h"].size(0) for e in enc)
    K = max(len(e["markers"]) for e in enc)
    d = enc[0]["h"].size(1)
    dtype = next(parts["scorer"].parameters()).dtype
    h = torch.zeros((len(enc), L, d), device=dev, dtype=dtype)
    att = torch.zeros((len(enc), L), device=dev, dtype=torch.bool)
    mpos = torch.zeros((len(enc), K), device=dev, dtype=torch.long)
    mmask = torch.zeros((len(enc), K), device=dev, dtype=torch.bool)
    for j, e in enumerate(enc):
        n = e["h"].size(0)
        h[j, :n] = e["h"].to(dtype)
        att[j, :n] = True
        mpos[j, :len(e["markers"])] = torch.tensor(e["markers"], device=dev)
        mmask[j, :len(e["markers"])] = True
    qtype = torch.tensor([e["qtype"] for e in enc], device=dev)
    h = h + parts["type_emb"](qtype)[:, None, :]
    for layer in parts["head"].layers:
        h = layer(h, src_key_padding_mask=~att)
    m = torch.gather(h, 1, mpos[:, :, None].expand(-1, -1, h.size(-1)))
    logits = parts["scorer"](m).squeeze(-1).float().masked_fill(~mmask, -1e4)
    return logits / temperature


def scores(parts: dict, enc: list[dict], temperature: float, batch: int = 32) -> list[list[float]]:
    import torch

    out = []
    was = parts["head"].training
    for p in parts.values():
        p.eval()
    with torch.no_grad():
        for i in range(0, len(enc), batch):
            probs = torch.softmax(head_logits(parts, enc[i:i + batch], temperature), -1)
            out += [row[:LEVELS].tolist() for row in probs.cpu()]
    for p in parts.values():
        p.train(was)
    return out
