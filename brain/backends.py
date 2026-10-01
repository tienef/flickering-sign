"""Fast (System 1) and slow (System 2) backends.

Fast: `appraise(state, verbs) -> {action, confidence, salience}` — one call per
tick. `state` holds natural-language fields for the model plus a `_ctx` dict of
structured context (stubs may read it; Laya must not see it).

Slow: `submit(workspace, t)` starts a deliberation without blocking;
`poll(t) -> [thought]` returns the ones that have finished (a payload's `_job` id
comes back as `job`, so the brain can drop a thought it has interrupted). A thought is
{goal, action?, expectation?, tokens, error?}. The body keeps acting while it thinks.

- StubFast / StubSlow: heuristic, deterministic, stdlib — scaffolding to
  exercise the loop. Their "goals" are canned; they do NOT count as evidence
  for the curiosity checklist.
- LayaFast: one Laya forward pass per tick answering every gating question
  (a value per action channel + salience; see its docstring for the two modes).
  CPU or CUDA (`brain.json` → `fast.device`).
- QwenSlow: an OpenAI-shaped call to the LiteLLM gateway on a background
  thread. Model, thinking, token budget and system prompt come from the
  bundle's `brain.json` → `slow`; URL and key from the genome, BRAIN_GATEWAY_URL / BRAIN_GATEWAY_KEY, or `.brain.env`.
- QwenFast: the three-level architecture's middle level (PLAN 1.1): the same LLM
  read in one pass, the probability of each option's letter; the same `appraise`
  shape, asked only on the ticks the fast level flags (brain.py). `brain.json` → `middle`.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

INTERNAL = ("rest", "sleep")


class StubFast:
    name = "stub-fast"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def appraise(self, state: dict, verbs: dict, mind: str | None = None) -> dict:
        ctx = state["_ctx"]
        urg = ctx["urgency"]
        novelty = ctx["novelty"]
        r = self.rng.random()
        world_verbs = [v for v in verbs if v not in INTERNAL]

        if urg.get("sleep_pressure", 0.0) > 0.6 and "sleep" in verbs:
            act, conf = "sleep", 0.85
        elif urg.get("hunger", 0.0) > 0.7 and "rest" in verbs and r < 0.5:
            act, conf = "rest", 0.7
        elif ctx.get("goal_verb") in verbs and r < 0.7:
            act, conf = ctx["goal_verb"], 0.75                 # follow the goal line
        elif urg.get("boredom", 0.0) > 0.3:
            restless = [v for v in world_verbs if v != "wait"] or world_verbs
            act = self.rng.choice(restless)
            conf = 0.5 - 0.35 * urg["boredom"]                 # restless and unsure
        else:
            act = self.rng.choice(world_verbs)
            conf = 0.35 + 0.5 * (1.0 - novelty) * (0.6 + 0.4 * self.rng.random())
        conf = max(0.0, min(1.0, conf))
        return {"action": act, "confidence": conf, "salience": novelty}


class StubSlow:
    """Answers `latency` ticks after submission with a canned goal: try the
    verb it has used least. Deterministic under a seed."""

    name = "stub-slow"

    def __init__(self, seed: int = 0, latency: int = 3):
        self.rng = random.Random(seed + 1)
        self.latency = latency
        self._queue: list[tuple[int, str, dict]] = []

    @property
    def busy(self) -> bool:
        return bool(self._queue)

    def submit(self, payload: dict, t: int, kind: str = "thought") -> None:
        self._queue.append((t + self.latency, kind, payload))

    def poll(self, t: int) -> list[dict]:
        due = [(k, p) for when, k, p in self._queue if when <= t]
        self._queue = [q for q in self._queue if q[0] > t]
        jobs = {"thought": self._think, "consolidation": self._consolidate, "review": self._review,
                "idle": self._idle, "cues": self._cues}
        return [dict(jobs[k](p), kind=k, job=p.get("_job")) for k, p in due]

    def close(self) -> None:
        pass

    def _consolidate(self, job: dict) -> dict:
        """One note per thing mentioned in the episodes: the latest thing seen of it."""
        latest = {}
        for ep in job["episodes"]:
            m = re.search(r"at the (\w+)", ep["text"])
            if m:
                latest[m.group(1)] = ep["text"]
        notes = [{"title": f"The {k}", "text": v} for k, v in latest.items()]
        return {"notes": notes, "contradictions": [], "resolved": [],
                "tokens": self.rng.randint(800, 2000)}

    def _review(self, job: dict) -> dict:
        """A note held if its first word shows up in the day's episodes, else 1 in 3 fail."""
        day = " ".join(e["text"].lower() for e in job["episodes"])
        verdicts = []
        for n in job["notes"]:
            word = (n["title"].split() or ["?"])[-1].lower()
            v = "held" if word in day else ("failed" if self.rng.random() < 0.34 else "untested")
            verdicts.append({"title": n["title"], "verdict": v})
        return {"verdicts": verdicts, "tokens": self.rng.randint(300, 900)}

    def _think(self, ws: dict) -> dict:
        counts = ws["verb_counts"]
        world_verbs = [v for v in ws["verbs"] if v not in INTERNAL]
        verb = min(world_verbs, key=lambda v: (counts.get(v, 0), self.rng.random()))
        heard = [p for p in ws["percepts"] if " said: " in p or " is here" in p]
        words = None
        if "speak" in ws["verbs"] and heard and self.rng.random() < 0.6:
            verb, words = "speak", f"I hear you. I am trying to {min(world_verbs)} things here."
        task = ws.get("task") or ("find a way out" if "task" in ws else None)
        if ws.get("motive") == "boredom":
            goal = f"what happens if I {verb}?"
        else:
            goal = f"keep going; try to {verb}"
        return {"goal": goal, "action": verb, **({"words": words} if words else {}),
                **({"task": task, "task_done": self.rng.random() < 0.1} if task else {}),
                "expectation": "something I have not seen yet",
                "checked": bool(ws.get("last_expectation")),
                "expectation_met": self.rng.random() < 0.5 if ws.get("last_expectation") else None,
                "tokens": self.rng.randint(300, 1200)}

    def _idle(self, ws: dict) -> dict:
        """A daydream: the oldest memory in the workspace, turned over."""
        about = (ws.get("memories") or ws.get("recent") or ["nothing much"])[0]
        return {"thought": f"that time: {about}"[:200], "tokens": self.rng.randint(200, 800)}

    def _cues(self, job: dict) -> dict:
        """One moment per note: its title's last word in front of me; the line is its first sentence;
        with "What my acts did" (v4), an act drawn from the body's actions or none."""
        acts = list(job["verbs"]) + ["none"]
        cues = [{"title": n["title"], "when": f"In front of me: {(n['title'].split() or ['it'])[-1].lower()}",
                 "recall": n["text"].split(". ")[0],
                 **({"act": self.rng.choice(acts)} if job.get("did") is not None else {})} for n in job["notes"]]
        return {"cues": cues, "tokens": self.rng.randint(300, 900)}


def _softmax(values: dict, temperature: float) -> dict:
    t = max(temperature, 1e-6)
    top = max(values.values())
    w = {k: math.exp((v - top) / t) for k, v in values.items()}
    z = sum(w.values())
    return {k: x / z for k, x in w.items()}


class LayaFast:
    """Laya as System 1: thalamus (salience) and basal ganglia (action selection)
    in ONE forward pass. ~1.3 s per pass on the box CPU in `values` mode.
    Import is lazy so stubs need no torch.

    Two modes (`brain.json` → `fast.mode`):
    - `values` (default): one score question per action channel — "how much would
      this answer how the mind feels and move it toward its goal?" — then the
      action is SAMPLED from a softmax over those values at `choice_temperature`
      (a knob: dopamine explores, stress exploits). Confidence = the chosen
      action's share of the competition at a fixed reference temperature, so close
      values mean conflict (the anterior cingulate's signal).
    - `choice`: one multiple-choice question. Measured on the box: it lets the
      passive verbs win whatever the mind feels, and locks onto its own last
      action (`wait` at 1.00 after "I waited") — perseveration.

    Habits (step 5c, `habits.py`): each mind may have its own decision head (the striatum),
    trained during its sleep; `attach` loads it, `appraise(mind=...)` uses it, `train_head`
    trains a copy. The encoder (the cortex) is shared and never trained.
    """

    name = "laya"

    ACTION = ("Given what this mind senses, how it feels and what it is aiming "
              "for, what does it do next?")
    VALUE = "This mind could {desc}. How much would that answer how it feels and move it toward its goal?"
    VALUE_LEGEND = ["not at all", "somewhat", "very well"]
    SALIENCE = ("How much does this moment deserve the mind's attention?",
                ["nothing worth noticing", "somewhat interesting", "striking, demands attention"])
    CONFIDENCE_TEMPERATURE = 0.15   # on the 0..2 value scale; fixed, so confidence is comparable over time

    def __init__(self, device: str = "cpu", mode: str = "values", seed: int = 0,
                 checkpoints=("english",), half: bool = False, salience: bool = True):
        from laya import Router  # lazy: only on the box
        if mode not in ("values", "choice"):
            raise ValueError(f"unknown fast.mode {mode!r} (expected 'values' or 'choice')")
        self.salience = salience
        # Only the checkpoints the brain uses: preload=True would hold all three
        # (~2.3 GB) for nothing.
        checkpoints = list(checkpoints)
        self.checkpoints = checkpoints
        self.mode = mode
        self.name = f"laya:{mode}"
        # Build on the CPU, then (for cuda) convert and move: the card never holds
        # fp32 weights, and PyTorch's cache is emptied, so the sidecar stays small
        # (measured 2026-09-24: loading on cuda then halving held 2.2 GiB).
        self.router = Router(preload=False, device="cpu", max_loaded=len(checkpoints))
        self.router.preload(checkpoints)
        if device != "cpu":
            import torch
            try:
                for name in checkpoints:
                    agent = self.router.load(name)
                    if half:
                        agent.model.half()                 # weights in fp16: ~half the VRAM
                    agent.model.to(device)
                    agent.device = torch.device(device)
                self.router.device = device
                torch.cuda.empty_cache()
            except RuntimeError as e:                      # no room on the card: stay on CPU
                print(f"[laya] could not move to {device} ({str(e)[:120]}); staying on CPU")
                self._to_cpu()
        self.half = half
        self._agent = self.router.load(checkpoints[0])
        from .habits import HEAD_PARTS
        self._base_parts = {k: getattr(self._agent.model, k) for k in HEAD_PARTS}
        self.heads: dict[str, dict] = {}                   # mind -> its own decision head (step 5c)
        self.device = str(self._agent.device)
        self.name = f"laya:{mode}@{self.device}" + (":fp16" if half and self.device != "cpu" else "")
        self.rng = random.Random(seed)
        self.calls = 0
        self.seconds = 0.0

    def _to_cpu(self) -> None:
        import torch
        for name in self.checkpoints:
            agent = self.router.load(name)
            agent.model.float().to("cpu")
            agent.device, agent.dtype = torch.device("cpu"), torch.float32
        self.router.device = "cpu"
        for parts in getattr(self, "heads", {}).values():
            for m in parts.values():
                m.float().to("cpu")
        self.device = "cpu"
        self.name = f"laya:{self.mode}@cpu"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def appraise(self, state: dict, verbs: dict, mind: str | None = None) -> dict:
        parts = self.heads.get(mind) if mind else None
        if not parts:
            return self._appraise(state, verbs)
        model = self._agent.model
        for k, m in parts.items():
            setattr(model, k, m)                            # this mind's striatum on the shared cortex
        try:
            return self._appraise(state, verbs)
        finally:
            for k, m in self._base_parts.items():
                setattr(model, k, m)

    def _appraise(self, state: dict, verbs: dict) -> dict:
        public = {k: v for k, v in state.items() if not k.startswith("_")}
        instr, legend = self.SALIENCE
        questions = {"salience": {"type": "score", "instructions": instr, "criteria": legend}}
        if self.mode == "choice" and not self.salience:
            questions = {}                                  # the fast head: the choice alone (the gate says what is salient)
        if self.mode == "values":
            for v, desc in verbs.items():
                questions["value:" + v] = {"type": "score", "criteria": self.VALUE_LEGEND,
                                           "instructions": self.VALUE.format(desc=desc)}
        else:
            questions["action"] = {"type": "choice", "instructions": self.ACTION, "criteria": dict(verbs)}
        t0 = time.monotonic()
        try:
            answers = self.router.predict(public, questions)["answers"]
        except RuntimeError as e:
            # Laya's own CUDA-error fallback moves the model to the CPU but leaves
            # half weights there; make them fp32 and retry once on the CPU.
            if "dtype" not in str(e) and "CUDA" not in str(e):
                raise
            print(f"[laya] {str(e)[:120]} -> falling back to CPU fp32")
            self._to_cpu()
            answers = self.router.predict(public, questions)["answers"]
        self.seconds += time.monotonic() - t0
        self.calls += 1
        salience = float(answers["salience"].get("score", 0.0)) / (len(legend) - 1) if "salience" in answers else 0.0

        if self.mode == "choice":
            act = answers["action"]
            probs = act.get("probabilities", {})
            choice = act.get("choice")
            if choice not in verbs:
                choice = max((v for v in probs if v in verbs), key=probs.get, default="rest")
            return {"action": choice, "salience": salience,
                    "confidence": float(act.get("confidence", probs.get(choice, 0.0))),
                    "probabilities": {k: round(p, 3) for k, p in probs.items()}}

        values = {v: float(answers["value:" + v].get("score", 0.0)) for v in verbs}
        raw = {k: round(x, 4) for k, x in values.items()}   # before the intention's pull: what habits learn on
        ctx = state.get("_ctx", {})
        if ctx.get("goal_bias") and ctx.get("goal_verb") in values:
            # the intention's pull (step 10g): acting on it is also less of a conflict
            values[ctx["goal_verb"]] += float(ctx["goal_bias"])
        for v, b in (ctx.get("cue_bias") or {}).items():
            if v in values:
                values[v] += float(b)                       # food in sight, felt by the hungry (step 10x)
        for v, b in (ctx.get("act_bias") or {}).items():
            if v in values:
                values[v] += float(b)                       # the act a recalled cue names (v4 cues)
        temperature = float(ctx.get("choice_temperature", 0.12))
        probs = _softmax(values, temperature)
        r, acc, choice = self.rng.random(), 0.0, None
        for v, p in probs.items():
            acc += p
            if r <= acc:
                choice = v
                break
        choice = choice or max(probs, key=probs.get)
        confidence = _softmax(values, self.CONFIDENCE_TEMPERATURE)[choice]
        return {"action": choice, "salience": salience, "confidence": confidence,
                "probabilities": {k: round(p, 3) for k, p in probs.items()},
                "values": {k: round(x, 3) for k, x in values.items()}, "raw_values": raw}


    # -- habits (step 5c) ---------------------------------------------------
    def _value_q(self, desc: str) -> dict:
        return {"type": "score", "criteria": self.VALUE_LEGEND, "instructions": self.VALUE.format(desc=desc)}

    def _salience_q(self) -> dict:
        return {"type": "score", "instructions": self.SALIENCE[0], "criteria": self.SALIENCE[1]}

    def _inference_copy(self, parts: dict) -> dict:
        import copy
        ref = next(self._base_parts["scorer"].parameters())
        return {k: copy.deepcopy(m).to(device=ref.device, dtype=ref.dtype).eval() for k, m in parts.items()}

    def attach(self, mind: str, head_path) -> bool:
        """Load a mind's own head if it has one (else it uses the original)."""
        from pathlib import Path
        if not Path(head_path).exists():
            return False
        from safetensors.torch import load_file
        parts = self._inference_copy(self._base_parts)
        weights = load_file(str(head_path))
        for k, m in parts.items():
            m.load_state_dict({n[len(k) + 1:]: t for n, t in weights.items() if n.startswith(k + ".")})
        self.heads[mind] = parts
        return True

    def train_head(self, mind: str, examples: list, anchors: list, probe: list, cfg: dict, rng, head_path) -> dict:
        """One night: train a copy of the mind's head on the examples (their targets) and the
        anchors (the original head's answers), check the probe against the original, keep the
        new head or roll back."""
        import copy
        import time as _time
        import torch
        from . import habits as H

        t0 = _time.monotonic()
        agent = self._agent
        temp = float(agent.temperature_by_options.get("score:3-5", agent.temperature[1]))
        base = self._base_parts
        current = self.heads.get(mind, base)
        teach = [(e["state"], self._value_q(e["desc"])) for e in examples]
        anchor_items = []
        for r in anchors:
            pick = rng.sample(sorted(r["verbs"]), min(int(cfg["anchor_verbs"]), len(r["verbs"])))
            anchor_items.append((r["state"], self._salience_q()))
            anchor_items += [(r["state"], self._value_q(r["verbs"][v])) for v in pick]
        probe_items = []
        for r in probe:
            probe_items.append((r["state"], self._salience_q()))
            probe_items += [(r["state"], self._value_q(d)) for d in r["verbs"].values()]
        cache = getattr(self, "enc_cache", None)          # offline replays only (deploy/probe/habits_offline.py)
        enc_t = H.encode(agent, teach, cache=cache)
        enc_a = H.encode(agent, anchor_items, cache=cache)
        enc_p = H.encode(agent, probe_items, cache=cache)
        t_enc = _time.monotonic() - t0
        anchor_targets = H.scores(base, enc_a, temp)
        probe_base = [H._expected(p) for p in H.scores(base, enc_p, temp)]
        probe_before = [H._expected(p) for p in H.scores(current, enc_p, temp)]
        teach_before = [H._expected(p) for p in H.scores(current, enc_t, temp)]

        parts = {k: copy.deepcopy(m).float().train() for k, m in current.items()}
        params = [p for m in parts.values() for p in m.parameters()]
        opt = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=0.0)
        data = ([(enc_t[i], e["target"], float(e.get("w", 1.0))) for i, e in enumerate(examples)]
                + [(enc_a[i], anchor_targets[i], float(cfg["anchor_weight"])) for i in range(len(enc_a))])
        bs, steps, losses = int(cfg["batch"]), 0, []
        for _ in range(int(cfg["epochs"])):
            order = list(range(len(data)))
            rng.shuffle(order)
            for i in range(0, len(order), bs):
                chunk = [data[j] for j in order[i:i + bs]]
                logits = H.head_logits(parts, [c[0] for c in chunk], temp)
                logp = torch.log_softmax(logits, -1)[:, :H.LEVELS]
                tgt = torch.tensor([c[1] for c in chunk], device=logp.device, dtype=logp.dtype)
                w = torch.tensor([c[2] for c in chunk], device=logp.device, dtype=logp.dtype)
                loss = -((tgt * logp).sum(-1) * w).sum() / w.sum()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                losses.append(loss.item())
                steps += 1
                if steps >= int(cfg["max_steps"]):
                    break
            if steps >= int(cfg["max_steps"]):
                break
        teach_after = [H._expected(p) for p in H.scores(parts, enc_t, temp)]
        probe_after = [H._expected(p) for p in H.scores(parts, enc_p, temp)]

        def mean(xs):
            xs = list(xs)
            return round(sum(xs) / len(xs), 4) if xs else None

        goal = [H._expected(e["target"]) for e in examples]
        drift = mean(abs(a - b) for a, b in zip(probe_after, probe_base))
        out = {"examples": len(examples), "steps": steps, "encode_s": round(t_enc, 2),
               "loss_first": round(losses[0], 4) if losses else None,
               "loss_last": round(sum(losses[-5:]) / len(losses[-5:]), 4) if losses else None,
               "gap_before": mean(abs(b - g) for b, g in zip(teach_before, goal)),
               "gap_after": mean(abs(a - g) for a, g in zip(teach_after, goal)),
               # how far the taught values rose (System 2's choices; what came before a good moment)
               "moved": {k: mean(a - b for a, b, e in zip(teach_after, teach_before, examples)
                                 if e["kind"] == k and (k == "distill" or e.get("z", 0) > 0))
                         for k in ("distill", "reinforce", "vary")},
               "drift_before": mean(abs(a - b) for a, b in zip(probe_before, probe_base)),
               "drift": drift, "accepted": drift is None or drift <= float(cfg["max_drift"])}
        if out["accepted"]:
            from safetensors.torch import save_file
            self.heads[mind] = self._inference_copy(parts)
            save_file({f"{k}.{n}": t.detach().to(torch.float16).cpu().contiguous()
                       for k, m in parts.items() for n, t in m.state_dict().items()}, str(head_path))
        else:
            out["why"] = f"probe drift {drift} > {cfg['max_drift']}: rolled back"
        del enc_t, enc_a, enc_p, parts, opt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out

    def distill_head(self, mind: str, examples: list, probe: list, cfg: dict, rng, head_path) -> dict:
        """One night of the fast head (three levels): train a copy of the mind's head so that its answer to
        the choice question matches the middle level's probabilities over the acts (cross-entropy to the soft
        labels), judge it on the held-out labels (agreement with the middle level's top act, cross-entropy)
        against the head it replaces, keep it or roll back. Examples and probe: {state, verbs, target, w}."""
        import copy
        import time as _time
        import torch
        from laya.common import QTYPES, temp_bucket
        from . import habits as H

        t0 = _time.monotonic()
        agent = self._agent
        current = self.heads.get(mind, self._base_parts)

        def items(rows):
            return [(r["state"], {"type": "choice", "instructions": self.ACTION, "criteria": dict(r["verbs"])})
                    for r in rows]

        def temps(rows):
            qt = QTYPES["choice"]
            return [float(agent.temperature_by_options.get(temp_bucket(qt, len(r["verbs"])), agent.temperature[qt]))
                    for r in rows]

        enc_t, enc_p = H.encode(agent, items(examples)), H.encode(agent, items(probe))
        t_enc = _time.monotonic() - t0

        def logp(parts, enc, T):
            z = H.head_logits(parts, enc, 1.0) / torch.tensor(T, device=enc[0]["h"].device)[:, None]
            return torch.log_softmax(z, -1)

        def target(rows, K, dev):
            y = torch.zeros((len(rows), K), device=dev)
            for j, r in enumerate(rows):
                t = torch.tensor(r["target"][:K], device=dev)
                y[j, :len(t)] = t / max(float(t.sum()), 1e-6)
            return y

        def judge(parts, enc, rows, T):
            if not rows:
                return None, None
            agree, ce = [], []
            for p in parts.values():
                p.eval()
            with torch.no_grad():
                for i in range(0, len(rows), 32):
                    lp = logp(parts, enc[i:i + 32], T[i:i + 32])
                    y = target(rows[i:i + 32], lp.size(1), lp.device)
                    ce += (-(y * lp).sum(-1)).tolist()
                    agree += (lp.argmax(-1) == y.argmax(-1)).float().tolist()
            return round(sum(agree) / len(agree), 4), round(sum(ce) / len(ce), 4)

        Tt, Tp = temps(examples), temps(probe)
        before = judge(current, enc_p, probe, Tp)
        parts = {k: copy.deepcopy(m).float().train() for k, m in current.items()}
        params = [p for m in parts.values() for p in m.parameters()]
        opt = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=0.0)
        bs, steps, losses = int(cfg["batch"]), 0, []
        for _ in range(int(cfg["epochs"])):
            order = list(range(len(examples)))
            rng.shuffle(order)
            for i in range(0, len(order), bs):
                idx = order[i:i + bs]
                lp = logp(parts, [enc_t[j] for j in idx], [Tt[j] for j in idx])
                y = target([examples[j] for j in idx], lp.size(1), lp.device)
                w = torch.tensor([float(examples[j].get("w", 1.0)) for j in idx], device=lp.device)
                loss = (-(y * lp).sum(-1) * w).sum() / w.sum()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                losses.append(loss.item())
                steps += 1
                if steps >= int(cfg["max_steps"]):
                    break
            if steps >= int(cfg["max_steps"]):
                break
        after = judge(parts, enc_p, probe, Tp)
        taught = judge(parts, enc_t, examples, Tt)
        out = {"examples": len(examples), "steps": steps, "encode_s": round(t_enc, 2),
               "loss_first": round(losses[0], 4) if losses else None,
               "loss_last": round(sum(losses[-5:]) / len(losses[-5:]), 4) if losses else None,
               "agree_taught": taught[0], "agree_before": before[0], "agree_after": after[0],
               "ce_before": before[1], "ce_after": after[1]}
        out["accepted"] = before[0] is None or after[0] >= before[0] - float(cfg["max_worse"])
        if out["accepted"]:
            from safetensors.torch import save_file
            self.heads[mind] = self._inference_copy(parts)
            save_file({f"{k}.{n}": t.detach().to(torch.float16).cpu().contiguous()
                       for k, m in parts.items() for n, t in m.state_dict().items()}, str(head_path))
        else:
            out["why"] = f"held-out agreement {after[0]} < {before[0]} - {cfg['max_worse']}: rolled back"
        del enc_t, enc_p, parts, opt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out


def _read_env_file(path: str) -> dict:
    out = {}
    try:
        with open(os.path.expanduser(path), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    out[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return out


_THINK = re.compile(r"<think>.*?</think>", re.S)


def _last_json_object(text: str) -> dict | None:
    """The last parseable TOP-LEVEL {...} in a reply (models sometimes wrap or
    preface it). Scans forward and skips past each object it parses, so an
    object nested inside the answer (a note inside {"notes": [...]}) is never
    mistaken for the answer itself."""
    text = _THINK.sub("", text or "")
    found, i = None, text.find("{")
    while i != -1:
        depth, end = 0, -1
        for j in range(i, len(text)):
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            if depth == 0:
                end = j
                break
        obj = None
        if end != -1:
            try:
                obj = json.loads(text[i:end + 1])
            except ValueError:
                pass
        if isinstance(obj, dict):
            found = obj
            i = text.find("{", end + 1)
        else:
            i = text.find("{", i + 1)
    return found


def _obj(**props) -> dict:
    """A strict JSON-schema object: every property required, no other."""
    return {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}


def _list(items: dict) -> dict:
    return {"type": "array", "items": items}


def _nul(t: str) -> dict:
    return {"type": [t, "null"]}


class QwenSlow:
    """Qwen as System 2 through the gateway, one deliberation at a time on a
    background thread. The system prompt is part of the bundle's genome
    (`brain.json` → `slow.system_prompt`): it shapes what the mind takes itself
    to be, so it is a variable of the experiment, not plumbing."""

    name = "qwen"

    def __init__(self, cfg: dict):
        env = _read_env_file(cfg.get("env_file", ".brain.env"))
        # three levels: `url` = the engine's own port (the middle level's model, thinking), no key
        self.url = cfg.get("url") or os.environ.get("BRAIN_GATEWAY_URL") or env.get("BRAIN_GATEWAY_URL")
        self.key = None if cfg.get("url") else (os.environ.get("BRAIN_GATEWAY_KEY") or env.get("BRAIN_GATEWAY_KEY"))
        if not self.url or not (self.key or cfg.get("url")):
            raise RuntimeError("no endpoint for the slow level: slow.url, BRAIN_GATEWAY_URL (and BRAIN_GATEWAY_KEY) or .brain.env")
        self.model = cfg.get("model", "qwen-fp8")
        self.think = bool(cfg.get("think", True))
        # q27 thinks only with a budget (--request-think); vLLM ignores the budget key.
        self.think_budget = cfg.get("think_budget")
        # vLLM ignores a thinking budget; Qwen3.8's template reads `reasoning_effort` (P0: low = ~417 tokens, 10.8 s)
        self.reasoning_effort = cfg.get("reasoning_effort")
        self.json_schema = bool(cfg.get("json_schema", False))     # vLLM's structured outputs (not q27)
        # the night's jobs (review, consolidation, cues) with or without thinking: on vLLM a consolidation thought
        # over 60 episodes ran past 4,000 tokens (104 s) with no reply; without thinking 6.6 s (P1 step 7)
        self.think_at_night = bool(cfg.get("think_at_night", True))
        self.max_tokens = int(cfg.get("max_tokens", 2000))
        # the effort of a thought, chosen by the brain per thought (3n I6: `slow.effort`): a cap per effort (the
        # thinking and the reply share max_tokens: at medium one thought ran past 2,000 and had no reply)
        self.effort_max_tokens = {k: int(v) for k, v in ((cfg.get("effort") or {}).get("max_tokens") or {}).items()}
        self.timeout = float(cfg.get("timeout_s", 300))
        self.system = cfg.get("system_prompt", "")
        self.consolidation_system = cfg.get("consolidation_prompt", "")
        self.consolidation_max_tokens = int(cfg.get("consolidation_max_tokens", 4000))
        self.review_system = cfg.get("review_prompt", "")         # step 10b: notes as hypotheses
        self.idle_system = cfg.get("idle_prompt", "")             # step 10ad: a thought nobody asked for
        self.cue_system = cfg.get("cue_prompt", "")               # recall by cues written at sleep
        self.name = f"qwen:{self.model}" + ("" if self.think else ":nothink")
        # two workers: an interrupting thought (step 10d) starts while the one it
        # supersedes finishes; without an `attention` genome only one runs at a time
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="system2")
        self._pending = []

    @property
    def busy(self) -> bool:
        return bool(self._pending)

    def warm(self, timeout: float = 480.0) -> float:
        """One tiny call so the gateway's lease hook swaps the card to this model
        BEFORE anything else (Laya) goes on it. Returns the seconds it took."""
        body = {"model": self.model, "max_tokens": 1,
                "messages": [{"role": "user", "content": "ready?"}],
                "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(
            self.url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
            headers={**({"Authorization": "Bearer " + self.key} if self.key else {}), "Content-Type": "application/json"})
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
        return round(time.monotonic() - t0, 1)

    def submit(self, payload: dict, t: int, kind: str = "thought") -> None:
        job = {"thought": self._deliberate, "consolidation": self._consolidate,
               "review": self._review, "idle": self._daydream, "cues": self._cues}[kind]
        self._pending.append(self._pool.submit(lambda: dict(job(payload), kind=kind, job=payload.get("_job"))))

    def poll(self, t: int) -> list[dict]:
        done = [f for f in self._pending if f.done()]
        self._pending = [f for f in self._pending if not f.done()]
        return [f.result() for f in done]

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def render(ws: dict) -> str:
        lines = ["Right now: " + " ".join(ws["percepts"])]
        if ws.get("recent"):
            lines.append("Just before, oldest first:")
            lines += [f"- {r}" for r in ws["recent"]]
        if ws.get("memories"):
            lines.append("Memories that come to mind:")
            lines += [f"- {m}" for m in ws["memories"]]
        if ws.get("knowledge"):
            lines.append("What the mind knows:")
            lines += [f"- {k}" for k in ws["knowledge"]]
        if ws.get("open"):
            lines.append("Things that do not add up:")
            lines += [f"- {c}" for c in ws["open"]]
        if ws.get("expectations"):
            lines.append("What the mind has learned to expect from each action here:")
            lines += [f"- {e}" for e in ws["expectations"]]
        le = ws.get("last_expectation")
        if le:
            lines.append(f"Last time, the mind expected: {le['text']}")
            lines.append("What happened since: " + (" / ".join(le["outcomes"]) or "nothing yet"))
        lines.append("Feeling: " + ("; ".join(ws["felt"]) if ws.get("felt")
                                    else ws.get("felt_neutral", "calm and content")))
        if ws.get("stuck"):                                # 3n I2: what keeps failing, counted
            lines.append("What keeps failing: " + ws["stuck"])
        if ws.get("urge"):                                 # step 10r: System 1's urge to speak
            lines.append(ws["urge"])
        if "task" in ws:                                   # step 10g genomes: goals by horizon
            lines.append("Current task: " + (ws.get("task") or "none"))
        lines.append("Current goal: " + (ws.get("goal") or "none"))
        lines.append("Available actions:")
        lines += [f"- {v}: {d}" for v, d in ws["verbs"].items()]
        return "\n".join(lines)

    @staticmethod
    def render_consolidation(job: dict) -> str:
        lines = ["Experiences since the last sleep, oldest first:"]
        lines += [f"- (t={e['t']}) {e['text']}" + (f" [felt: {e['feeling']}]" if e.get("feeling") else "")
                  for e in job["episodes"]]
        lines.append("")
        if job.get("notes"):
            lines.append("Notes the mind already keeps:")
            for n in job["notes"]:
                lines += [f"## {n['title']}", n["text"]]
        else:
            lines.append("The mind keeps no notes yet.")
        if job.get("open"):
            lines += ["", "Things that did not add up before:"] + [f"- {c}" for c in job["open"]]
        if job.get("traces"):
            lines += ["", "What struck the mind in the moment today (first impressions, from single moments):"]
            lines += [f"- {t}" for t in job["traces"]]
        return "\n".join(lines)

    @staticmethod
    def render_review(job: dict) -> str:
        lines = ["Experiences since the last sleep, oldest first:"]
        lines += [f"- (t={e['t']}) {e['text']}" for e in job["episodes"]]
        lines += ["", "Notes the mind keeps:"]
        for n in job["notes"]:
            lines += [f"## {n['title']}", n["text"]]
        return "\n".join(lines)

    @staticmethod
    def render_cues(job: dict) -> str:
        lines = ["Notes the mind keeps:"]
        for n in job["notes"]:
            lines += [f"## {n['title']}", n["text"]]
        lines += ["", "What the body can do:"] + [f"- {v}: {d}" for v, d in job["verbs"].items()]
        if job.get("did") is not None:          # v4: what each act did, from the mind's own episodes
            lines += ["", "What my acts did (from my memories):"] + job["did"]
        return "\n".join(lines)

    def _post(self, system: str, user: str, max_tokens: int, temperature: float,
              think_budget: int | None = None, schema: dict | None = None,
              night: bool = False, effort: str | None = None) -> tuple[dict | None, dict]:
        """One request. `effort` (3n I6): "none" asks without thinking, "low" / "medium" set reasoning_effort for
        this call, None keeps the genome's `reasoning_effort`. A thought whose thinking used up the cap before any
        reply (finish_reason length) is asked once more without thinking, rather than lost."""
        if effort is not None and effort != "none" and effort in self.effort_max_tokens:
            max_tokens = self.effort_max_tokens[effort]
        body = {"model": self.model, "max_tokens": max_tokens, "temperature": temperature,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}]}
        if schema and self.json_schema:
            # the reply's shape imposed by the engine after the thinking (owner 2026-09-30: code, not the prompt,
            # guarantees the JSON): no reply without JSON, and acts only from the body's actions
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "reply", "schema": schema, "strict": True}}
        think = self.think and (self.think_at_night or not night) and effort != "none"
        budget = think_budget or self.think_budget
        if think and (effort or self.reasoning_effort):
            body["reasoning_effort"] = effort or self.reasoning_effort
        if not think:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        elif budget:
            body["chat_template_kwargs"] = {"enable_thinking": True, "thinking_budget": int(budget)}
            body["thinking_token_budget"] = int(budget)
        req = urllib.request.Request(
            self.url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={**({"Authorization": "Bearer " + self.key} if self.key else {}), "Content-Type": "application/json"})
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                d = json.load(r)
        except Exception as e:  # network, HTTP 4xx/5xx, a swap that outlasted the timeout
            return None, {"error": f"{type(e).__name__}: {e}"[:300], "tokens": 0,
                          "seconds": round(time.monotonic() - t0, 2)}
        choice = (d.get("choices") or [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content") or ""
        if think and choice.get("finish_reason") == "length" and not (content or "").strip():
            # the thinking used up the cap: no reply to shape (not a malformed JSON); ask again at once, unthinking
            # an unthinking reply is ~100-300 tokens: a floor, whatever cap the thinking had
            obj, out = self._post(system, user, max(max_tokens, 600), temperature, schema=schema, night=night,
                                  effort="none")
            out.update(cut_thinking=True, seconds=round(out.get("seconds", 0) + time.monotonic() - t0, 2),
                       tokens=out.get("tokens", 0) + int((d.get("usage") or {}).get("total_tokens", 0)))
            return obj, out
        # the inner monologue, for scoring only (it is never fed back to the mind)
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        if not reasoning and "</think>" in content:
            reasoning, content = content.split("</think>", 1)
            reasoning = reasoning.replace("<think>", "")
        out = {"tokens": int((d.get("usage") or {}).get("total_tokens", 0)),
               "seconds": round(time.monotonic() - t0, 2),
               "raw": content.strip()[:1000]}
        if effort is not None:
            out["effort"] = effort
        if reasoning.strip():
            out["reasoning"] = reasoning.strip()[:3000]
        obj = _last_json_object(content)
        if obj is None:
            out["error"] = "no JSON object in reply"
        return obj, out

    def _deliberate(self, ws: dict) -> dict:
        # an orienting thought (step 10e) may carry a smaller budget: a direction, not an essay
        obj, out = self._post(self.system, self.render(ws), self.max_tokens,
                              ws.get("temperature") or 0.7, think_budget=ws.get("_think_budget"),
                              effort=ws.get("_effort"),
                              schema=_obj(task=_nul("string"), task_done=_nul("boolean"), goal={"type": "string"},
                                          action={"enum": list(ws["verbs"]) + [None]}, words=_nul("string"),
                                          expectation=_nul("string"), expectation_met=_nul("boolean")))
        if obj is None:
            return out
        action = obj.get("action")
        met = obj.get("expectation_met")
        words = obj.get("words")
        if (isinstance(action, str) and action not in ws["verbs"] and "speak" in ws["verbs"]
                and re.match(r"(?i)\s*(speak|say)\b", action)):
            # the words written into the action itself: "speak: Who is calling?"
            words = words or re.sub(r"(?i)^\s*(speak|say)\b\s*[:,-]?\s*", "", action)
            action = "speak"
        if action == "speak" and words:
            out["words"] = str(words).strip()[:240]            # shared worlds: what to say aloud
        done = obj.get("task_done")
        out.update(task=(str(obj["task"]).strip() if obj.get("task") else None),
                   task_done=done if isinstance(done, bool) else str(done).lower() == "true",
                   goal=(str(obj["goal"]).strip() if obj.get("goal") else None),
                   action=action if action in ws["verbs"] else None,
                   expectation=obj.get("expectation"),
                   checked=bool(ws.get("last_expectation")),
                   expectation_met=met if isinstance(met, bool) else None)
        return out

    def _daydream(self, ws: dict) -> dict:
        """Step 10ad: the same workspace, a prompt that says nothing asks for System 2, and a free
        line back instead of a goal and an action."""
        obj, out = self._post(self.idle_system or self.system, self.render(ws), self.max_tokens,
                              ws.get("temperature") or 0.7, think_budget=ws.get("_think_budget"),
                              schema=_obj(thought={"type": "string"}))
        if obj is None:
            return out
        out["thought"] = str(obj["thought"]).strip()[:400] if obj.get("thought") else None
        return out

    def _consolidate(self, job: dict) -> dict:
        obj, out = self._post(self.consolidation_system, self.render_consolidation(job),
                              self.consolidation_max_tokens, 0.3, night=True,
                              schema=_obj(notes=_list(_obj(title={"type": "string"}, text={"type": "string"})),
                                          contradictions=_list({"type": "string"}), resolved=_list({"type": "string"})))
        if obj is None:
            return out
        if not isinstance(obj.get("notes"), list):
            out["error"] = "reply has no notes list"     # never close out episodes on a bad reply
            return out
        notes = [n for n in obj["notes"] if isinstance(n, dict)]
        out.update(notes=notes,
                   contradictions=[str(c) for c in obj.get("contradictions") or []],
                   resolved=[str(c) for c in obj.get("resolved") or []])
        return out


    def _review(self, job: dict) -> dict:
        obj, out = self._post(self.review_system, self.render_review(job),
                              self.consolidation_max_tokens, 0.2, night=True,
                              schema=_obj(verdicts=_list(_obj(title={"type": "string"},
                                                              verdict={"enum": ["held", "failed", "untested"]}))))
        if obj is None:
            return out
        if not isinstance(obj.get("verdicts"), list):
            out["error"] = "reply has no verdicts list"
            return out
        out["verdicts"] = [v for v in obj["verdicts"] if isinstance(v, dict)]
        return out

    def _cues(self, job: dict) -> dict:
        """At sleep, after the consolidation: the moments each touched note will matter, and a line to
        recall then (the consolidation's settings, as the offline bench's `cues` phase)."""
        cue = dict(title={"type": "string"}, when={"type": "string"}, recall={"type": "string"})
        if job.get("did") is not None:
            cue["act"] = {"enum": list(job["verbs"]) + ["none"]}
        obj, out = self._post(self.cue_system, self.render_cues(job), self.consolidation_max_tokens, 0.3,
                              night=True, schema=_obj(cues=_list(_obj(**cue))))
        if obj is None:
            return out
        if not isinstance(obj.get("cues"), list):
            out["error"] = "reply has no cues list"
            return out
        out["cues"] = [c for c in obj["cues"] if isinstance(c, dict)]
        return out


class QwenFast:
    """The middle level (PLAN 1.1): the LLM read in ONE pass, no thinking. The mind's state as text, the
    question, the actions as lettered options; the reply's first token is read as the probability of each
    option's letter (openjev's readout, on an untuned model: no extra weights). The act is drawn from those
    probabilities at the `choice_temperature` knob (1 = the model's own odds; dopamine explores, stress
    exploits); the code biases the brain may add (the intention's pull, a food cue, a cue's act) are added to
    the log-probabilities, default 0 in the three-level template. Confidence = the drawn act's probability,
    so a split vote is the conflict signal the arbiter hears.

    Benched on 2026-09-30 (`probe/one_model.py`, PLAN 1.1): lures not followed, the intention followed
    (0.78-0.80 on top), feelings steer, rules generalised to unseen kinds; 0.13 s a call alone.

    Genome (`brain.json` -> `middle`): model, url (a vLLM server, e.g. http://127.0.0.1:8000/v1; else
    the gateway from `env_file`), system_prompt, question, timeout_s. It gives no salience: the fast level
    says what is salient (brain.py, the gate)."""

    LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    SYSTEM = ("You are the fast, intuitive part of a mind. You do not deliberate: you read what the mind senses, how "
              "it feels and what it is aiming for, and answer at once with what it does next. Reply with the letter "
              "of one option and nothing else.")
    QUESTION = "Given what this mind senses, how it feels and what it is aiming for, what does it do next?"
    TOP = 20                                          # vLLM's default max_logprobs

    def __init__(self, cfg: dict, seed: int = 0):
        env = _read_env_file(cfg.get("env_file", ".brain.env"))
        self.url = cfg.get("url") or os.environ.get("BRAIN_GATEWAY_URL") or env.get("BRAIN_GATEWAY_URL")
        # a direct vLLM port needs no key; the gateway does
        self.key = None if cfg.get("url") else (os.environ.get("BRAIN_GATEWAY_KEY") or env.get("BRAIN_GATEWAY_KEY"))
        if not self.url:
            raise RuntimeError("no URL for the middle level: middle.url, BRAIN_GATEWAY_URL or .brain.env")
        self.model = cfg.get("model", "qwen38-27b-uncensored")
        self.system = cfg.get("system_prompt", self.SYSTEM)
        self.question = cfg.get("question", self.QUESTION)
        self.timeout = float(cfg.get("timeout_s", 60))
        self.name = f"qwenfast:{self.model}"
        self.rng = random.Random(seed)
        self.calls = 0
        self.errors = 0
        self.seconds = 0.0

    @staticmethod
    def render(state: dict) -> str:
        """The state as the benches rendered it: one line per field the model may see."""
        return "\n".join(f"{k[:1].upper() + k[1:]}: {v}" for k, v in state.items() if v and not k.startswith("_"))

    def prompt(self, state: dict, names: list[str], verbs: dict) -> str:
        return (self.render(state) + "\n\n" + self.question + "\nOptions:\n"
                + "\n".join(f"{self.LETTERS[i]}. {n}: {verbs[n]}" for i, n in enumerate(names))
                + "\nAnswer with one letter.")

    def _post(self, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        req = urllib.request.Request(self.url.rstrip("/") + "/chat/completions",
                                     data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)

    def logprobs(self, state: dict, verbs: dict) -> tuple[dict, dict]:
        """{verb: log-probability of its letter} at the first token (letters outside the top ones are
        absent), and the call's facts (seconds, prompt tokens)."""
        names = list(verbs)[:len(self.LETTERS)]
        body = {"model": self.model, "max_tokens": 1, "temperature": 0, "logprobs": True,
                "top_logprobs": self.TOP, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "system", "content": self.system},
                             {"role": "user", "content": self.prompt(state, names, verbs)}]}
        t0 = time.monotonic()
        try:
            d = self._post(body)
        except Exception:  # noqa: BLE001 - one retry: a busy engine, a dropped connection
            d = self._post(body)
        secs = time.monotonic() - t0
        self.calls += 1
        self.seconds += secs
        top = d["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        letter = {self.LETTERS[i]: n for i, n in enumerate(names)}
        lp = {}
        for t in top:
            tok = t["token"].strip()
            if tok in letter and letter[tok] not in lp:
                lp[letter[tok]] = float(t["logprob"])
        return lp, {"seconds": round(secs, 3), "tokens": int((d.get("usage") or {}).get("prompt_tokens", 0))}

    def appraise(self, state: dict, verbs: dict, mind: str | None = None) -> dict:
        try:
            lp, facts = self.logprobs(state, verbs)
        except Exception as e:  # noqa: BLE001 - the brain carries on without the middle level
            self.errors += 1
            return {"action": None, "confidence": None, "error": f"{type(e).__name__}: {e}"[:300]}
        if not lp:
            self.errors += 1
            return {"action": None, "confidence": None, "error": "no option letter in the first token", **facts}
        mass = sum(math.exp(x) for x in lp.values())    # how much of the reply went to an option at all
        floor = min(lp.values()) - 2.0                   # an option outside the top letters: below all of them
        logits = {v: lp.get(v, floor) for v in verbs}
        probs = _softmax(logits, 1.0)                    # the model's own odds over the options
        ctx = state.get("_ctx", {})
        biased = dict(logits)
        if ctx.get("goal_bias") and ctx.get("goal_verb") in biased:
            biased[ctx["goal_verb"]] += float(ctx["goal_bias"])
        for key in ("cue_bias", "act_bias"):
            for v, b in (ctx.get(key) or {}).items():
                if v in biased:
                    biased[v] += float(b)
        draw = _softmax(biased, float(ctx.get("choice_temperature", 1.0)))
        r, acc, choice = self.rng.random(), 0.0, None
        for v, p in draw.items():
            acc += p
            if r <= acc:
                choice = v
                break
        choice = choice or max(draw, key=draw.get)
        return {"action": choice, "confidence": probs[choice], "top": max(probs, key=probs.get),
                "probabilities": {k: round(p, 3) for k, p in probs.items()},
                "mass": round(mass, 3), **facts}


def make_fast(name: str, *, seed: int = 0, cfg: dict | None = None):
    cfg = cfg or {}
    if name == "none":                                # three levels without a head yet: the middle level appraises
        return None
    if name == "stub":
        return StubFast(seed=seed)
    if name == "laya":
        return LayaFast(device=cfg.get("device", "cpu"), mode=cfg.get("mode", "values"), seed=seed,
                        checkpoints=cfg.get("checkpoints", ["english"]), half=bool(cfg.get("half", False)),
                        salience=bool(cfg.get("salience", True)))
    raise ValueError(f"unknown fast backend: {name!r} (expected 'stub' or 'laya')")


def make_middle(name: str, *, seed: int = 0, cfg: dict | None = None):
    """The middle level (three-level genomes): `qwen` (QwenFast) or `stub` (StubFast's heuristics)."""
    if name == "stub":
        return StubFast(seed=seed)
    if name == "qwen":
        return QwenFast(cfg or {}, seed=seed)
    raise ValueError(f"unknown middle backend: {name!r} (expected 'stub' or 'qwen')")


def make_slow(name: str, *, seed: int = 0, cfg: dict | None = None):
    if name == "stub":
        return StubSlow(seed=seed)
    if name == "qwen":
        return QwenSlow(cfg or {})
    raise ValueError(f"unknown slow backend: {name!r} (expected 'stub' or 'qwen')")
