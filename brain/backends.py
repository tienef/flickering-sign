"""Fast (System 1) and slow (System 2) backends.

Fast: `appraise(state, verbs) -> {action, confidence, salience}` — one call per
tick. `state` holds natural-language fields for the model plus a `_ctx` dict of
structured context (stubs may read it; Laya must not see it).

Slow: `submit(workspace, t)` starts a deliberation without blocking;
`poll(t) -> [thought]` returns the ones that have finished. A thought is
{goal, action?, expectation?, tokens, error?}. The body keeps acting while it thinks.

- StubFast / StubSlow: heuristic, deterministic, stdlib — scaffolding to
  exercise the loop. Their "goals" are canned; they do NOT count as evidence
  for the curiosity checklist.
- LayaFast: one Laya forward pass per tick answering every gating question
  (a value per action channel + salience; see its docstring for the two modes).
  CPU or CUDA (`brain.json` → `fast.device`).
- QwenSlow (backend name `qwen` or `openai`): an OpenAI-compatible
  `/chat/completions` call on a background thread — any server works (vLLM,
  llama.cpp, Ollama, LiteLLM, a hosted API). Model, thinking, token budget and
  system prompt come from the bundle's `brain.json` → `slow`; URL and key from
  BRAIN_GATEWAY_URL / BRAIN_GATEWAY_KEY, or from the file named by
  `slow.env_file` (default `.brain.env`). Built and tested with Qwen; the
  thinking switches it sends are Qwen's and other servers ignore them.
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

    def appraise(self, state: dict, verbs: dict) -> dict:
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
        return [dict(self._think(p) if k == "thought" else self._consolidate(p), kind=k)
                for k, p in due]

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

    def _think(self, ws: dict) -> dict:
        counts = ws["verb_counts"]
        world_verbs = [v for v in ws["verbs"] if v not in INTERNAL]
        verb = min(world_verbs, key=lambda v: (counts.get(v, 0), self.rng.random()))
        heard = [p for p in ws["percepts"] if " said: " in p or " is here" in p]
        words = None
        if "speak" in ws["verbs"] and heard and self.rng.random() < 0.6:
            verb, words = "speak", f"I hear you. I am trying to {min(world_verbs)} things here."
        if ws.get("motive") == "boredom":
            goal = f"what happens if I {verb}?"
        else:
            goal = f"keep going; try to {verb}"
        return {"goal": goal, "action": verb, **({"words": words} if words else {}),
                "expectation": "something I have not seen yet",
                "checked": bool(ws.get("last_expectation")),
                "expectation_met": self.rng.random() < 0.5 if ws.get("last_expectation") else None,
                "tokens": self.rng.randint(300, 1200)}


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
                 checkpoints=("english",), half: bool = False):
        from laya import Router  # lazy: only on the box
        if mode not in ("values", "choice"):
            raise ValueError(f"unknown fast.mode {mode!r} (expected 'values' or 'choice')")
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
        self.device = str(self.router.load(checkpoints[0]).device)
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
        self.device = "cpu"
        self.name = f"laya:{self.mode}@cpu"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def appraise(self, state: dict, verbs: dict) -> dict:
        public = {k: v for k, v in state.items() if not k.startswith("_")}
        instr, legend = self.SALIENCE
        questions = {"salience": {"type": "score", "instructions": instr, "criteria": legend}}
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
        salience = float(answers["salience"].get("score", 0.0)) / (len(legend) - 1)

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
        temperature = float(state.get("_ctx", {}).get("choice_temperature", 0.12))
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
                "values": {k: round(x, 3) for k, x in values.items()}}


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


class QwenSlow:
    """Qwen (or any OpenAI-compatible model) as System 2, one deliberation at a time on a
    background thread. The system prompt is part of the bundle's genome
    (`brain.json` → `slow.system_prompt`): it shapes what the mind takes itself
    to be, so it is a variable of the experiment, not plumbing."""

    name = "qwen"

    def __init__(self, cfg: dict):
        env = _read_env_file(cfg.get("env_file", ".brain.env"))
        self.url = os.environ.get("BRAIN_GATEWAY_URL") or env.get("BRAIN_GATEWAY_URL")
        # A local server usually needs no key; any placeholder is sent then.
        self.key = os.environ.get("BRAIN_GATEWAY_KEY") or env.get("BRAIN_GATEWAY_KEY") or "none"
        if not self.url:
            raise RuntimeError("no System 2 endpoint: set BRAIN_GATEWAY_URL (e.g. http://localhost:8000/v1) "
                               "and BRAIN_GATEWAY_KEY if it needs one, or put them in .brain.env")
        self.model = cfg.get("model", "qwen-fp8")
        self.think = bool(cfg.get("think", True))
        # q27 thinks only with a budget (--request-think); vLLM ignores the budget key.
        self.think_budget = cfg.get("think_budget")
        self.max_tokens = int(cfg.get("max_tokens", 2000))
        self.timeout = float(cfg.get("timeout_s", 300))
        self.system = cfg.get("system_prompt", "")
        self.consolidation_system = cfg.get("consolidation_prompt", "")
        self.consolidation_max_tokens = int(cfg.get("consolidation_max_tokens", 4000))
        self.name = f"qwen:{self.model}" + ("" if self.think else ":nothink")
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="system2")
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
            headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
        return round(time.monotonic() - t0, 1)

    def submit(self, payload: dict, t: int, kind: str = "thought") -> None:
        job = self._deliberate if kind == "thought" else self._consolidate
        self._pending.append(self._pool.submit(lambda: dict(job(payload), kind=kind)))

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
        return "\n".join(lines)

    def _post(self, system: str, user: str, max_tokens: int, temperature: float) -> tuple[dict | None, dict]:
        body = {"model": self.model, "max_tokens": max_tokens, "temperature": temperature,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}]}
        if not self.think:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        elif self.think_budget:
            body["chat_template_kwargs"] = {"enable_thinking": True,
                                            "thinking_budget": int(self.think_budget)}
            body["thinking_token_budget"] = int(self.think_budget)
        req = urllib.request.Request(
            self.url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                d = json.load(r)
        except Exception as e:  # network, HTTP 4xx/5xx, a swap that outlasted the timeout
            return None, {"error": f"{type(e).__name__}: {e}"[:300], "tokens": 0,
                          "seconds": round(time.monotonic() - t0, 2)}
        msg = (d.get("choices") or [{}])[0].get("message", {})
        content = msg.get("content") or ""
        # the inner monologue, for scoring only (it is never fed back to the mind)
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        if not reasoning and "</think>" in content:
            reasoning, content = content.split("</think>", 1)
            reasoning = reasoning.replace("<think>", "")
        out = {"tokens": int((d.get("usage") or {}).get("total_tokens", 0)),
               "seconds": round(time.monotonic() - t0, 2),
               "raw": content.strip()[:1000]}
        if reasoning.strip():
            out["reasoning"] = reasoning.strip()[:3000]
        obj = _last_json_object(content)
        if obj is None:
            out["error"] = "no JSON object in reply"
        return obj, out

    def _deliberate(self, ws: dict) -> dict:
        obj, out = self._post(self.system, self.render(ws), self.max_tokens,
                              ws.get("temperature") or 0.7)
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
        out.update(goal=(str(obj["goal"]).strip() if obj.get("goal") else None),
                   action=action if action in ws["verbs"] else None,
                   expectation=obj.get("expectation"),
                   checked=bool(ws.get("last_expectation")),
                   expectation_met=met if isinstance(met, bool) else None)
        return out

    def _consolidate(self, job: dict) -> dict:
        obj, out = self._post(self.consolidation_system, self.render_consolidation(job),
                              self.consolidation_max_tokens, 0.3)
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


def make_fast(name: str, *, seed: int = 0, cfg: dict | None = None):
    cfg = cfg or {}
    if name == "stub":
        return StubFast(seed=seed)
    if name == "laya":
        return LayaFast(device=cfg.get("device", "cpu"), mode=cfg.get("mode", "values"), seed=seed,
                        checkpoints=cfg.get("checkpoints", ["english"]), half=bool(cfg.get("half", False)))
    raise ValueError(f"unknown fast backend: {name!r} (expected 'stub' or 'laya')")


def make_slow(name: str, *, seed: int = 0, cfg: dict | None = None):
    if name == "stub":
        return StubSlow(seed=seed)
    if name in ("qwen", "openai"):
        return QwenSlow(cfg or {})
    raise ValueError(f"unknown slow backend: {name!r} (expected 'stub', 'qwen' or 'openai')")
