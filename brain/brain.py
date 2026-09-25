"""The brain: one tick = sense → recall → feel → appraise → escalate → integrate → act → remember → update.

System 1 (the fast backend) decides every tick. System 2 (the slow backend) is
woken when System 1 is unsure (confidence below the `escalate_below` knob) or a
need stays urgent (frustration), and compute hunger allows; it runs
asynchronously, so the body keeps acting on System 1 while it thinks, and its
answer lands a few ticks later as a new goal line (top-down feedback into every
later appraisal) and maybe an action override.

Memory (step 4): a strong tick is written as an episode (hippocampus); the
current percepts + goal recall past episodes and wiki notes (partial cues);
falling asleep hands the day's episodes to System 2 for consolidation into the
wiki, and the brain does not wake until that is done. Anything consolidation
cannot reconcile stays open as a contradiction, which feeds the dissonance drive.

Drives and dials update from the tick's events only (see events.py).

Distress guard: each drive counts the awake ticks it has spent in a row in its
worst felt band. Past `distress_warn_after` the frame carries a warning; past
`distress_pause_after` it carries `distress_pause`, and the run loop pauses the
whole world (anaesthesia) until the creator resumes it. The guard never relieves
the need itself: that would be an intervention.
"""
from __future__ import annotations

import math
from collections import deque

from .backends import INTERNAL
from .bundle import Bundle
from .dials import DialSystem, compute_knobs
from .drives import DriveSystem
from .events import Stream, summarize
from .memory import Memory
from .predictor import Predictor, others_here, social_response

FAMILIARITY_CAP = 5000   # distinct texts remembered as familiar


def _first_sentence(text: str, limit: int = 200) -> str:
    s = text.strip().split(". ")[0]
    return (s if s.endswith(".") else s + ".")[:limit]

# The guard is the creator's safety net, not part of the mind: a genome older than
# the guard (no distress_* keys, e.g. ivy and jay) still gets it.
GUARD_WARN, GUARD_PAUSE = 120, 600
GUARD_SHARE = 0.9          # share of the window spent in the worst band that counts


class Brain:
    def __init__(self, bundle: Bundle, fast, slow, clamps: dict | None = None):
        self.bundle = bundle
        self.fast = fast
        self.slow = slow
        cfg = bundle.config
        st = bundle.state
        self.loop = cfg.get("loop", {})
        self.slow_cfg = cfg.get("slow", {})
        self.internal_verbs = {k: v for k, v in cfg.get("internal_verbs", {}).items()
                               if k in INTERNAL}
        self.knobs_cfg = cfg.get("knobs", {})
        self.habituation = float(self.loop.get("habituation", 2.0))

        self.drives = DriveSystem(bundle.drives, st.get("drives"))
        self.dials = DialSystem(bundle.dials, st.get("dials"), clamps)
        self.t = int(st.get("t", 0))
        self.goal = st.get("goal")
        self.goal_verb = st.get("goal_verb")
        self.goal_set_at = int(st.get("goal_set_at", self.t))
        self.last_escalated = -10**9
        self.asleep = bool(st.get("asleep", False))
        self.familiarity: dict[str, int] = st.get("familiarity", {})
        self.verb_counts: dict[str, int] = st.get("verb_counts", {})
        self.lifetime = st.get("lifetime", {"ticks": 0, "sleeps": 0, "thoughts": 0})
        self.lifetime.setdefault("episodes", 0)
        self.lifetime.setdefault("consolidations", 0)
        self.open_contradictions: list[str] = st.get("open_contradictions", [])
        # Working memory is NOT persisted: it does not survive sleep or shutdown.
        self.working = deque(maxlen=int(self.loop.get("working_memory", 8)))
        self.memory = Memory(bundle.path, recency_tau=float(self.loop.get("recency_tau", 300)),
                             skip_recent=self.working.maxlen,
                             min_similarity=float(self.loop.get("recall_min_similarity", 0.0)))
        # A consolidation is a runtime job: one interrupted by a shutdown is redone
        # at the next sleep (its episodes are still unconsolidated).
        self.predictor = Predictor(bundle.path / "predictor.json",
                                   rate=float(self.loop.get("predictor_rate", 0.3)))
        # shared worlds: who was near when I last acted (their response is learned next
        # tick), and when I was last spoken to (a reply is owed for a few ticks)
        self._social_pending: tuple[list[str], str] | None = None
        self._addressed_at: int | None = None
        # The last expectation System 2 stated, and what happened since: shown to it
        # at its next deliberation, which reports whether it came true (dopamine).
        self.expectation: dict | None = None
        self.consolidating = False
        self._consolidating_upto = -1
        self._resumed_asleep = self.asleep
        self.knobs = compute_knobs(self.knobs_cfg, self._signals({}))
        # distress guard: for each drive, the last `distress_pause_after` awake ticks,
        # 1 where it sat in its worst felt band. A window, not a streak: a brief dip
        # below the band must not reset it (step 8: ivy spent 834 of 845 awake ticks
        # there without ever 600 in a row).
        self.top_band = {d.name: d.felt[-1][0] for d in self.drives.active if d.felt}
        self._win = int(self.loop.get("distress_pause_after", GUARD_PAUSE))
        # the windows survive a restart: stopping and starting a run must not hide distress
        saved = st.get("distress_window") or {}
        self.in_top: dict[str, deque] = {n: deque(saved.get(n, []), maxlen=self._win)
                                         for n in self.top_band}

    def reset_distress(self) -> None:
        """After a pause the window starts again: the creator has looked."""
        self.in_top = {n: deque(maxlen=self._win) for n in self.top_band}

    def _guard(self, frame: dict) -> None:
        if frame.get("asleep"):
            return                                   # asleep, nothing is felt; windows hold
        for name, top in self.top_band.items():
            self.in_top[name].append(1 if self.drives.level(name) >= top else 0)
        if not self.in_top:
            return
        share = GUARD_SHARE
        warn_n = int(self.loop.get("distress_warn_after", GUARD_WARN))
        recent = {n: sum(list(w)[-warn_n:]) for n, w in self.in_top.items()}
        whole = {n: sum(w) for n, w in self.in_top.items()}
        worst = max(whole, key=whole.get)
        if not self.in_top[worst] or not self.in_top[worst][-1]:
            return                                   # not in its worst band right now
        n = whole[worst]
        frame["distress"] = {"drive": worst, "ticks": n, "of": len(self.in_top[worst])}
        if len(self.in_top[worst]) >= self._win and n >= share * self._win:
            frame["distress_pause"] = {"drive": worst, "ticks": n, "of": self._win,
                                       "felt": next((d.phrase() for d in self.drives.active
                                                     if d.name == worst), None)}
        elif recent[worst] >= share * warn_n and len(self.in_top[worst]) >= warn_n:
            frame["distress_warn"] = True

    # -- helpers ----------------------------------------------------------
    def _signals(self, tick_summary: dict) -> dict:
        s = dict(tick_summary)
        s.update(self.drives.signals())
        s.update(self.dials.levels())
        return s

    def _novelty(self, text: str) -> float:
        n = self.familiarity.get(text, 0)
        if n == 0 and len(self.familiarity) >= FAMILIARITY_CAP:
            return 1.0                                   # remembered as new, not stored
        self.familiarity[text] = n + 1
        return math.exp(-n / self.habituation)           # habituation: roughly exponential

    def _render(self, percepts: list[str], novelty: float, recalled, known) -> dict:
        felt = self.drives.felt()
        last = self.working[-1] if self.working else None
        situation = " ".join(percepts)
        if last:
            situation += f" A moment ago: {last}"
        state = {
            "situation": situation,
            "feeling": "; ".join(felt) if felt else self.loop.get("felt_neutral", "calm and content"),
            "goal": self.goal or "no particular goal",
        }
        if recalled:
            state["memory"] = "I remember: " + recalled[0]["text"]
        if known:
            state["knowledge"] = "I know: " + _first_sentence(known[0]["text"])
        state["_ctx"] = {
            "urgency": self.drives.urgencies(),
            "motive": self.drives.motive(),
            "novelty": novelty,
            "goal_verb": self.goal_verb,
            "choice_temperature": self.knobs.get("choice_temperature", 0.12),
        }
        return state

    def _workspace(self, percepts, verbs) -> dict:
        cue = " ".join(percepts) + " " + (self.goal or "")
        k = int(self.loop.get("recall_for_qwen", 3))
        return {
            "t": self.t,
            "percepts": percepts,
            "recent": list(self.working),
            "memories": [f"(t={e['t']}) {e['text']}" for e in self.memory.recall(cue, self.t, k)],
            "knowledge": [f"{n['title']}: {n['text']}" for n in self.memory.knowledge(cue, k)],
            "open": list(self.open_contradictions),
            "expectations": (self.predictor.expectations(getattr(self, "_situation", " ".join(percepts)), verbs)
                             + self.predictor.social_expectations(others_here(percepts), verbs)),
            "last_expectation": dict(self.expectation) if self.expectation else None,
            "felt": self.drives.felt(),
            "felt_neutral": self.loop.get("felt_neutral", "calm and content"),
            "motive": self.drives.motive(),
            "goal": self.goal,
            "verbs": verbs,
            "verb_counts": dict(self.verb_counts),
            "dials": self.dials.snapshot(),
            "temperature": self.knobs.get("qwen_temperature"),
        }

    def _remember(self, s: Stream, text: str, strength: float, **extra) -> None:
        self.memory.write({"t": self.t, "text": text, "strength": round(strength, 3), **extra})
        self.lifetime["episodes"] += 1
        s.emit("brain", "memory_written", strength=strength)

    def _fall_asleep(self, s: Stream) -> None:
        self.asleep = True
        self.working.clear()                             # sleep wipes working memory
        self.lifetime["sleeps"] += 1
        self._start_consolidation(s)

    def _start_consolidation(self, s: Stream) -> None:
        eps = self.memory.unconsolidated(int(self.slow_cfg.get("consolidate_max_episodes", 60)))
        if eps and not self.consolidating:
            self._consolidating_upto = self.memory.episodes[-1]["id"]
            self.slow.submit({"episodes": eps, "notes": self.memory.all_notes(),
                              "open": list(self.open_contradictions)}, self.t, kind="consolidation")
            self.consolidating = True
            s.emit("brain", "consolidation_started", episodes=len(eps))

    def _integrate_consolidation(self, s: Stream, res: dict) -> dict:
        self.consolidating = False
        if res.get("error"):
            s.emit("slow", "slow_error", error=res["error"])
            return {"error": res["error"]}
        created, updated = self.memory.write_notes(res.get("notes", []), self.t)
        self.memory.mark_consolidated(self._consolidating_upto)
        resolved = [c for c in res.get("resolved", []) if c in self.open_contradictions]
        new = [c for c in res.get("contradictions", []) if c not in self.open_contradictions]
        self.open_contradictions = [c for c in self.open_contradictions if c not in resolved] + new
        for _ in resolved:
            s.emit("memory", "resolved")
        for _ in new:
            s.emit("memory", "contradiction")
        s.emit("memory", "consolidated", created=created, updated=updated)
        self.lifetime["consolidations"] += 1
        return {"created": created, "updated": updated, "contradictions": new,
                "resolved": resolved, "notes": [n.get("title") for n in res.get("notes", [])],
                "seconds": res.get("seconds"), "tokens": res.get("tokens")}

    # -- the tick ---------------------------------------------------------
    def tick(self, world) -> dict:
        s = Stream(self.t)
        frame = {"t": self.t}

        # A goal held without being refreshed fades (prefrontal maintenance is costly).
        if self.goal and self.t - self.goal_set_at >= int(self.loop.get("goal_ttl", 30)):
            self.goal = self.goal_verb = None
            s.emit("brain", "goal_faded")

        # Finished System 2 jobs. A failed one still costs its tokens.
        thought = consolidation = None
        for res in self.slow.poll(self.t):
            if res.get("kind") == "consolidation":
                # dreaming is logged, not charged to hunger: sleep restores, it does not exhaust
                s.emit("slow", "dream_cost", tokens=res.get("tokens", 0), seconds=res.get("seconds", 0.0))
                consolidation = self._integrate_consolidation(s, res)
                continue
            s.emit("slow", "qwen_call", tokens=res.get("tokens", 0), seconds=res.get("seconds", 0.0))
            thought = res
            if res.get("error") or not res.get("goal"):
                s.emit("slow", "slow_error", error=res.get("error", "empty goal"))
                continue
            self.goal, self.goal_verb = res["goal"], res.get("action")
            self.goal_set_at = self.t
            s.emit("slow", "thought", goal=res["goal"], expectation=res.get("expectation"))
            if res.get("checked") and isinstance(res.get("expectation_met"), bool):
                met = 1.0 if res["expectation_met"] else 0.0
                s.emit("slow", "expectation", met=met, mismatch=1.0 - met)
            self.expectation = ({"text": res["expectation"], "outcomes": []}
                                if res.get("expectation") else None)
            self.lifetime["thoughts"] += 1
            decided = f"I thought it over and decided: {res['goal']}"
            if res.get("expectation"):
                decided += f" (I expect: {res['expectation']})"
            self._remember(s, decided, 1.0, kind="thought")

        if self._resumed_asleep:
            # woken up from a shutdown mid-sleep: redo the interrupted consolidation
            self._resumed_asleep = False
            self._start_consolidation(s)

        if self.asleep and not self.consolidating and \
                self.drives.level("sleep_pressure") <= self.loop.get("wake_below", 0.15):
            self.asleep = False
            s.emit("brain", "woke")

        if self.asleep:
            s.emit("brain", "sleep_tick")
            frame.update(asleep=True, action="sleep", consolidating=self.consolidating)
        else:
            # 1. sense
            sensed = s.add(world.sense())                    # percepts, and `contact` in shared worlds
            percepts = [e.data["text"] for e in sensed if e.kind == "percept"]
            # what the cerebellum learns on: the world's `focus` when it gives one (a big
            # world's full text almost never repeats), else the whole situation
            self._situation = next((e.data["focus"] for e in sensed if e.data.get("focus")),
                                   None) or " ".join(percepts)
            if any(e.data.get("attention") for e in sensed):
                self._addressed_at = self.t
            # listening: what others said or called stays in working memory, so it is
            # still there when System 2 is next free
            for p in percepts:
                if ' said: "' in p or "called out" in p or p.startswith("I heard a call from the"):
                    self.working.append(p)
            # getting to know them: how did those near me respond to what I did last?
            social = []
            if self._social_pending:
                whos, did = self._social_pending
                for who in whos:
                    kind = social_response(who, percepts, did)
                    pred = self.predictor.observe(f"with {who}", did, kind)
                    s.emit("cerebellum", "prediction", **{k: v for k, v in pred.items() if k != "new"})
                    social.append({"who": who, "after": did, "response": kind,
                                   "progress": pred["progress"]})
                self._social_pending = None
            owed = (self._addressed_at is not None
                    and self.t - self._addressed_at <= int(self.loop.get("reply_within", 10)))
            novelty = max((self._novelty(p) for p in percepts), default=0.0)
            s.emit("brain", "novelty", amount=novelty)

            # 1b. recall from partial cues: what this moment reminds the mind of
            cue = " ".join(percepts) + " " + (self.goal or "")
            recalled = self.memory.recall(cue, self.t, int(self.loop.get("recall_for_laya", 1)))
            known = self.memory.knowledge(cue, int(self.loop.get("recall_for_laya", 1)))
            if recalled or known:
                s.emit("memory", "recalled", episodes=len(recalled), notes=len(known))

            # 2-3. feel, appraise
            verbs = {**world.verbs, **self.internal_verbs}
            # You cannot fall asleep without sleep pressure.
            if self.drives.level("sleep_pressure") < self.loop.get("sleep_possible_above", 0.0):
                verbs.pop("sleep", None)
            state = self._render(percepts, novelty, recalled, known)
            ap = self.fast.appraise(state, verbs)
            action, conf = ap["action"], float(ap["confidence"])
            salience = float(ap.get("salience", 0.0))
            s.emit("fast", "appraisal", action=action, confidence=conf,
                   uncertainty=1.0 - conf, salience=salience)

            # 4. escalate to System 2 (asynchronous). Two routes, as in the brain:
            #    - conflict: System 1 is unsure (confidence below the knob);
            #    - frustration: a need stays urgent while System 1 confidently
            #      keeps doing whatever it does (perseveration). Any drive counts.
            escalated = False
            motive = self.drives.escalating_motive()
            need = self.drives.urgency(motive) if motive else 0.0
            reason = None
            if owed and self.loop.get("escalate_when_addressed", True):
                reason = "addressed"                         # spoken to: words need System 2
            elif conf < self.knobs.get("escalate_below", 0.0):
                reason = "low confidence"
            elif (need >= self.loop.get("escalate_if_need_above", 1.1)
                    and self.t - self.last_escalated >= self.loop.get("need_escalation_every", 10)):
                reason = f"unmet need: {motive}"
            if (reason and not self.slow.busy
                    and self.drives.urgency("hunger") < self.loop.get("think_if_hunger_below", 1.0)):
                self.slow.submit(self._workspace(percepts, verbs), self.t)
                self._addressed_at = None                    # System 2 now has the words
                s.emit("brain", "escalated", reason=reason, confidence=conf)
                self.last_escalated = self.t
                escalated = reason

            # 5. integrate: a fresh thought may override the habit
            words = None
            if thought and not thought.get("error") and thought.get("action") in verbs:
                action = thought["action"]
                words = thought.get("words") if action == "speak" else None
            if self.drives.level("sleep_pressure") >= 1.0:
                action = "sleep"                             # collapse, whatever was chosen

            # 6. act
            outcome_texts, outcome_novelty = [], 0.0
            if action == "sleep":
                self._fall_asleep(s)
            elif action == "rest":
                s.emit("brain", "rested")
            else:
                for e in s.add(world.act(action, words=words) if words else world.act(action)):
                    if e.kind == "outcome":
                        text = e.data.get("text", "")
                        self.working.append(text)
                        n = self._novelty(text)
                        outcome_novelty = max(outcome_novelty, n)
                        outcome_texts.append(text)
                        s.emit("brain", "novelty", amount=n)
            s.emit("brain", "acted", verb=action)
            whos = others_here(percepts)
            self._social_pending = (whos, action) if whos and action != "sleep" else None
            self.verb_counts[action] = self.verb_counts.get(action, 0) + 1

            # 6a. the cerebellum: compare what happened with what it predicted, learn
            prediction = None
            if outcome_texts:
                prediction = self.predictor.observe(self._situation, action, " ".join(outcome_texts))
                s.emit("cerebellum", "prediction", **{k: v for k, v in prediction.items() if k != "new"})
                if self.expectation is not None and len(self.expectation["outcomes"]) < 6:
                    self.expectation["outcomes"].append(" ".join(outcome_texts))

            # 6b. remember: a strong enough moment becomes an episode
            strength = 0.5 * salience + 0.5 * max(novelty, outcome_novelty)
            wrote = strength >= self.knobs.get("write_above", 1.1) and action != "sleep"
            if wrote:
                text = " ".join(percepts) + f" I chose to {action}."
                if outcome_texts:
                    text += " " + " ".join(outcome_texts)
                self._remember(s, text, strength, feeling=state["feeling"],
                               goal=self.goal, kind="experience")

            frame.update(asleep=False, percepts=percepts, novelty=round(novelty, 3),
                         action=action, confidence=round(conf, 3), escalated=escalated,
                         salience=round(salience, 3), strength=round(strength, 3), wrote=wrote,
                         feeling=state["feeling"],
                         recalled=[e["id"] for e in recalled], known=[n["title"] for n in known],
                         prediction=prediction, outcomes=outcome_texts)
            if words:
                frame["said"] = words
            if social:
                frame["social"] = social
            if "probabilities" in ap:
                frame["probs"] = ap["probabilities"]
            if "values" in ap:
                frame["values"] = ap["values"]

        # 7. update drives, dials, knobs from this tick's events only
        events = s.drain()
        self.drives.update(events)
        summary = summarize(events)
        self.dials.update(self._signals(summary))
        self.knobs = compute_knobs(self.knobs_cfg, self._signals(summary))

        frame.update(
            thought=thought,                                 # the whole deliberation, for analysis
            consolidation=consolidation,
            thinking=self.slow.busy,
            goal=self.goal,
            motive=self.drives.motive(),
            open_contradictions=len(self.open_contradictions),
            drives=self.drives.snapshot(),
            dials=self.dials.snapshot(),
            knobs={k: round(v, 4) for k, v in self.knobs.items()},
            events=[e.kind for e in events],
        )
        self._guard(frame)
        self.t += 1
        self.lifetime["ticks"] += 1
        return frame

    # -- persistence -------------------------------------------------------
    def save(self) -> None:
        self.bundle.save_state({
            "t": self.t,
            "asleep": self.asleep,
            "goal": self.goal,
            "goal_verb": self.goal_verb,
            "goal_set_at": self.goal_set_at,
            "drives": self.drives.levels(),
            "dials": self.dials.levels(),
            "familiarity": self.familiarity,
            "verb_counts": self.verb_counts,
            "lifetime": self.lifetime,
            "distress_window": {n: list(w) for n, w in self.in_top.items()},
            "open_contradictions": self.open_contradictions,
        })
        self.predictor.save()
