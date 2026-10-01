"""The brain: one tick = sense → recall → feel → appraise → escalate → integrate → act → remember → update.

Three levels (PLAN 1.1; genomes with `fast.gate`): the fast level (code, `_fast_level`) decides each tick
whether the moment calls for a new choice; if it does, the middle level (`middle`: the LLM in one pass)
chooses, and the fast head (a per-mind Laya head, `fast`) proposes beside it and is distilled from those
choices at sleep; otherwise the body carries on. The slow level (`slow`: the same LLM thinking) is the
System 2 below, asked by the arbiter. Without `fast.gate`, the old two-system loop that follows:

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
With a review in the genome (step 10b), sleep first tests the notes against the
day and retires those that keep failing; only the survivors are consolidated.

Drives and dials update from the tick's events only (see events.py).

Distress guard: each drive counts the awake ticks it has spent in a row in its
worst felt band. Past `distress_warn_after` the frame carries a warning; past
`distress_pause_after` it carries `distress_pause`, and the run loop pauses the
whole world (anaesthesia) until the creator resumes it. The guard never relieves
the need itself: that would be an intervention.
"""
from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, deque

from .backends import INTERNAL
from .bundle import Bundle
from .dials import DialSystem, compute_knobs
from .drives import DriveSystem
from .events import Stream, clamp, summarize
from . import habits
from .memory import Memory, slug, tokens
from .predictor import Predictor, others_here, social_response

FAMILIARITY_CAP = 5000   # distinct texts remembered as familiar
# an experience episode reads "<percepts> I chose to <act>. <outcomes>" (written in tick(), step 6b)
_CHOSE = re.compile(r" I chose to (\S+?)\. (.+)$")


def _dedupe(items: list[dict], key: str) -> list[dict]:
    seen, out = set(), []
    for it in items:
        if it.get(key) not in seen:
            seen.add(it.get(key))
            out.append(it)
    return out


def _first_sentence(text: str, limit: int = 200) -> str:
    s = text.strip().split(". ")[0]
    return (s if s.endswith(".") else s + ".")[:limit]

# The guard is the creator's safety net, not part of the mind: a genome older than
# the guard (no distress_* keys, e.g. ivy and jay) still gets it.
GUARD_WARN, GUARD_PAUSE = 120, 600
GUARD_SHARE = 0.9          # share of the window spent in the worst band that counts


class Brain:
    def __init__(self, bundle: Bundle, fast, slow, clamps: dict | None = None, middle=None):
        self.bundle = bundle
        self.fast = fast
        self.slow = slow
        # three levels (PLAN 1.1): with `fast.gate` in the genome, the fast level (code) decides each tick whether
        # the moment calls for a new choice; only then is the middle level (`middle`, one LLM pass) asked, else
        # the body carries on. Without it, the old two-system loop: `fast` appraises every tick.
        self.middle = middle
        self.appraiser = middle or fast
        cfg = bundle.config
        self.gate = (cfg.get("fast") or {}).get("gate")
        self.carry = (cfg.get("fast") or {}).get("carry", {})
        self._prev = None                         # what the gate compares with: None = choose afresh (start, waking)
        self._last_act: dict | None = None        # the last act, whether it changed anything, how new its outcome was
        self._last_flag = int(bundle.state.get("t", 0))
        self._search_way: str | None = None       # the way a restless body searches (fast.carry.search)
        # traces written awake at a strong moment (genome `traces`, P3: learning within the day)
        tr = cfg.get("traces") or {}
        self.traces = tr if tr.get("on") else None
        self._trace_hist: deque = deque(maxlen=int(tr.get("chain_ticks", 6)) + 1)
        self._traces_today = 0
        self._gate_surprise = 0.0                # the cerebellum's surprise at the last act
        # frustration (3n I2, genome `stuck`): failed acts while something is wanted become `thwarted` events, which
        # the genome's drive (`stuck.drive`) integrates; what keeps failing is said to the slow level
        self.stuck = (cfg.get("stuck") or {}) if (cfg.get("stuck") or {}).get("on") else None
        self._stuck_last: tuple | None = None    # (act, what it met, t) of the last failed act
        self._stuck_run = {"act": None, "n": 0, "text": ""}
        # what relieves a felt need, in view (3n I5, genome `loop.need_cues`): the onset flags the gate
        self._relief_seen: dict = {}
        self._relief_onset = 0.0
        self._log: list[dict] = []                # this tick's changes in full text (P1 step 5: the inspector)
        self._last_recall: tuple = ([], [])
        st = bundle.state
        self.loop = cfg.get("loop", {})
        self.slow_cfg = cfg.get("slow", {})
        self.internal_verbs = {k: v for k, v in cfg.get("internal_verbs", {}).items()
                               if k in INTERNAL}
        self.knobs_cfg = cfg.get("knobs", {})
        self.habituation = float(self.loop.get("habituation", 2.0))

        self.drives = DriveSystem(bundle.drives, st.get("drives"))
        self.dials = DialSystem(bundle.dials, st.get("dials"), clamps)
        self.drives.felt_signals = self.dials.levels()
        # entrainment (genome `entrain`, 3k): the body learns the length of its world's day from the dark and
        # sets the pace of a drive (sleep pressure) so that a waking day fills it to `target` by nightfall
        en = cfg.get("entrain") or {}
        self.entrain = en if en.get("on") else None
        self._entrain = dict(st.get("entrain") or {"pace": float(en.get("init", 1.0)), "light": 0, "rise": 0.0,
                                                   "n": 0, "dark": False, "days": 0})
        if self.entrain:
            self.drives.set_pace(en.get("drive", "sleep_pressure"), self._entrain["pace"])
        self.t = int(st.get("t", 0))
        self.goal = st.get("goal")
        self.goal_verb = st.get("goal_verb")
        self.goal_set_at = int(st.get("goal_set_at", self.t))
        self.goal_motive = st.get("goal_motive")         # the drive the goal was thought under (3n Q1)
        # goals by horizon (step 10g): the goal line above is the *intention* (seconds, fades
        # after goal_ttl); the *task* lasts the waking period, until System 2 says it is done
        # or replaces it; sleep wipes it with working memory. Both live in state.json.
        self.task = st.get("task")
        # the last time the intention's action was done: (what was sensed, what came of it)
        self._last_follow: tuple[str, str] | None = None
        self.task_set_at = int(st.get("task_set_at", self.t))
        self.last_escalated = -10**9
        self.asleep = bool(st.get("asleep", False))
        self.familiarity: dict[str, int] = st.get("familiarity", {})
        self.verb_counts: dict[str, int] = st.get("verb_counts", {})
        self.lifetime = st.get("lifetime", {"ticks": 0, "sleeps": 0, "thoughts": 0})
        self.lifetime.setdefault("episodes", 0)
        self.lifetime.setdefault("consolidations", 0)
        self.open_contradictions: list[str] = st.get("open_contradictions", [])
        # effectance (step 10q): how often each effect was had (habituation), and the individual's
        # temperament, drawn once at birth from the genome's ranges and kept in its genome file
        self.effect_counts: dict[str, float] = st.get("effect_counts", {})
        # what fed me (step 10ab): the words of the outcomes that came with a meal, learnt,
        # so that hunger can bring back the memories of eating (state-dependent recall)
        self.food_words: list[str] = st.get("food_words", [])
        self.temperament = self._temperament(bundle, cfg)
        # Working memory is NOT persisted: it does not survive sleep or shutdown.
        self.working = deque(maxlen=int(self.loop.get("working_memory", 8)))
        self.memory = Memory(bundle.path, recency_tau=float(self.loop.get("recency_tau", 300)),
                             skip_recent=self.working.maxlen,
                             min_similarity=float(self.loop.get("recall_min_similarity", 0.0)))
        # A consolidation is a runtime job: one interrupted by a shutdown is redone
        # at the next sleep (its episodes are still unconsolidated).
        # Recall by cues written at sleep (PLAN "A vector index built at sleep"): notes come back
        # through an index of their cues, rebuilt from the wiki now and at every sleep.
        ri = self.loop.get("recall_index")
        if ri and ri.get("kind") == "cues":
            from .recall_index import CueIndex
            self.memory.cue_index = CueIndex(ri)
            self.memory.rebuild_index()
        # the body's actions as offered while awake: the cue writing at sleep lists them
        self.verbs_seen: dict[str, str] = st.get("verbs_seen", {})
        self.predictor = Predictor(bundle.path / "predictor.json",
                                   rate=float(self.loop.get("predictor_rate", 0.3)),
                                   prior=bool(self.loop.get("predictor_prior", False)))
        # shared worlds: who was near when I last acted (their response is learned next
        # tick), and when I was last spoken to (a reply is owed for a few ticks)
        self._social_pending: tuple[list[str], str] | None = None
        self._addressed_at: int | None = None
        # orienting response (step 10c): when my own action surprised me, System 2 is owed a look
        self._surprised_at: int | None = None
        self._focus_by_verb: dict[str, str] = {}      # an action's own focus, when the world gives one
        # attention (step 10d): sources bid for System 2, the thought in flight holds a level,
        # the arbiter (salience network) decides what interrupts it. Off without `attention`.
        self.attention = cfg.get("attention")
        self._bid: tuple[str, float, int] | None = None           # the strongest waiting bid
        self._thought_job: tuple[int, str, float, int] | None = None   # id, reason, priority, t
        self._job_seq = 0
        self._asked: dict[int, tuple[int, str | None]] = {}       # job -> (tick asked, motive then): 3n Q1
        self._last_need_bid = -10**9
        self._in_worst: set[str] = set()
        self._landed_at = -10**9                  # when the last thought gave a goal (step 10i's rest)
        self._free_ticks = 0                      # awake ticks System 2 has had nothing in flight (daydreams)
        # The last expectation System 2 stated, and what happened since: shown to it
        # at its next deliberation, which reports whether it came true (dopamine).
        self.expectation: dict | None = None
        self.consolidating = False
        self._consolidating_upto = -1
        # the cue writing in flight (10ae, owner: the mind wakes while it finishes): when it was
        # asked and the titles asked; a runtime job, lost at a shutdown (its notes are asked again)
        self._cues_job: tuple[int, list[str]] | None = None
        self._woke_at: int | None = None
        self._resumed_asleep = self.asleep
        self.knobs = compute_knobs(self.knobs_cfg, self._signals({}))
        # habits (step 5c): the day's appraisals are replayed in sleep and System 1's own head
        # learns from them. Off without a `habits` genome block.
        self.habits = habits.settings(cfg["habits"]) if cfg.get("habits") else None
        # the fast head (three levels, P1 step 3): a per-mind Laya head that proposes on the flagged ticks and is
        # distilled at sleep from the middle level's choices; off without `fast.head` or without a `fast` backend
        head = (cfg.get("fast") or {}).get("head")
        self.head = habits.head_settings(head) if head and self.gate and fast is not None else None
        if self.head:
            self.habits = None                    # the old reward habits (5c) would share, and clear, its day
        self._head_trust: deque = deque(st.get("head_trust", []), maxlen=int(self.head["trust_window"])
                                        if self.head else 1)
        self._head_night: dict | None = None
        self._rng = random.Random(f"{bundle.path.name}:{self.t}")
        self._habit_store = habits.Store(bundle.path / "habits") if (self.habits or self.head) else None
        self._mind = str(bundle.path.resolve())
        self._mind_kw = {}
        if ((self.habits and self.habits.get("train", True)) or self.head) and hasattr(fast, "attach"):
            fast.attach(self._mind, self._habit_store.head)
            self._mind_kw = {"mind": self._mind}
        self._habit_row: dict | None = None
        self._habits_night: dict | None = None
        # food cues (step 10x): the sight of food pulls System 1 toward it, the more the hungrier
        # (alliesthesia); the world says which acts lead to food (`food_by_verb`)
        self._food_seen = False
        self._searched_at = -10**9                # the last time the search verb was done (step 10ab)
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
                                       "felt": next((d.phrase(self.drives.felt_signals) for d in self.drives.active
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
        n = int(self.loop.get("recent_for_fast", 1))
        last = " ".join(list(self.working)[-n:]) if self.working else None
        situation = " ".join(percepts)
        if last:
            situation += f" A moment ago: {last}"
        state = {
            "situation": situation,
            "feeling": "; ".join(felt) if felt else self.loop.get("felt_neutral", "calm and content"),
            "goal": self.goal or "no particular goal",
        }
        if self.task:
            state["task"] = self.task
        if recalled:
            state["memory"] = "I remember: " + recalled[0]["text"]
        act = known[0].get("act") if known else None
        if known and ((self.loop.get("recall_index") or {}).get("laya_line") != "with_act" or act):
            # through the cue index: the line the sleep wrote to recall at such a moment. With laya_line
            # "with_act" (v4) only a cue that names an act reaches System 1 (Laya reads a line with no
            # act, mostly "X cannot be ...", as a pull toward X); System 2 still gets the note.
            state["knowledge"] = "I know: " + (known[0].get("recall") or _first_sentence(known[0]["text"]))
        state["_ctx"] = {
            "urgency": self.drives.urgencies(),
            "motive": self.drives.motive(),
            "novelty": novelty,
            "goal_verb": self.goal_verb,
            "choice_temperature": self.knobs.get("choice_temperature", 0.12),
            "goal_bias": self._goal_bias(),
        }
        if act and self.knobs.get("cue_act_pull"):
            # the act a recalled cue names pulls System 1 toward it, as the intention's does (v4): knowledge
            # biasing the basal ganglia without a thought. Whatever act the cue names; the `cue_act_pull` knob.
            state["_ctx"]["act_bias"] = {act: round(self.knobs["cue_act_pull"], 4)}
        return state

    def _spent(self, s: Stream, action: str, percepts, outcomes, frame: dict) -> None:
        """An intention ends when its action stops changing anything (step 10h): done
        again, the same situation gave the same outcome (a second bump into the closed
        door). The pull is released and System 2 is asked for the next step, as the basal
        ganglia let go of a failed action and the prefrontal cortex picks the next sub-goal."""
        if not self.attention or "intention_spent" not in self.attention.get("weights", {}):
            return
        if not self.goal_verb or action != self.goal_verb:
            return
        now = (" ".join(percepts), " ".join(outcomes))
        if now != self._last_follow:
            self._last_follow = now
            return
        s.emit("brain", "intention_spent", verb=action)
        frame["intention_spent"] = action
        self.goal_verb, self._last_follow = None, None       # the goal line stays, its pull goes
        self._offer(f"intention spent: {action}", self._weight("intention_spent"))

    def _lands_late(self, res: dict, asked: tuple[int, str | None] | None) -> bool:
        """A thought cut by a newer one still lands while its moment holds (3n Q1, genome `loop.intentions`):
        asked within `late_within` ticks, under the motive the mind still has, and no thought landed since."""
        late = int((self.loop.get("intentions") or {}).get("late_within", 0))
        return bool(late and asked and res.get("goal") and not res.get("error")
                    and self.t - asked[0] <= late and asked[1] == self.drives.motive()
                    and self._landed_at < asked[0])

    def _wake_intention(self, s: Stream, frame: dict) -> None:
        """At waking, the intention stored through the sleep (3n Q1): held afresh if its motive is still felt
        (`loop.intentions.keep_above`), else let go."""
        ic = self.loop.get("intentions") or {}
        if not ic.get("through_sleep") or not self.goal:
            return
        if self.goal_motive and self.drives.urgency(self.goal_motive) >= float(ic.get("keep_above", 0.5)):
            self.goal_set_at = self.t
            frame["intention_kept"] = self.goal
            s.emit("brain", "intention_kept")
        else:
            self.goal = self.goal_verb = None
            s.emit("brain", "goal_faded")

    def _goal_bias(self) -> float:
        """How much the intention pulls System 1 toward its verb (step 10g): the prefrontal
        cortex biasing the basal ganglia. The `goal_bias` knob, fading as the intention ages."""
        if not self.goal_verb or "goal_bias" not in self.knobs:
            return 0.0
        age = self.t - self.goal_set_at
        return round(self.knobs["goal_bias"] * 0.5 ** (age / float(self.loop.get("goal_bias_half_life", 15))), 4)

    def _food_cue(self, food_by_verb: dict, food_far: dict | None = None) -> dict:
        """Food in sight pulls System 1 toward the acts that lead to it, the more the hungrier
        (step 10x): alliesthesia (Cabanac 1971: the same food is more pleasant to the hungry) and
        incentive salience (Berridge: a hungry animal's dopamine makes food cues wanted; the
        lateral hypothalamus, orexin). No thought needed, as the goal's pull but keyed on the cue.
        Its onset, when hungry, also bids for System 2's attention (the cephalic phase: the sight
        of food readies the body). Genome: loop.food_cue {drive, gain, bid}.

        Appetitive search (step 10ab, owner's choice; genome keys `approach`, `search`): hunger
        drives foraging before food is in reach, as the lateral hypothalamus does (hungry animals
        search; Hoebel 1971, Jennings 2015). Food seen further away (`food_far`: the step toward
        it, from the world) pulls as the cue does, times `approach`; with nothing in sight, a hungry
        mind is pulled to its `search.verb` (look around) once every `search.every` ticks."""
        fc = self.loop.get("food_cue")
        # approach only while nothing is in reach: food at hand wins over food seen further away
        far = (food_far or {}) if fc and fc.get("approach") and not food_by_verb else {}
        seen = bool(food_by_verb) or bool(far)
        onset, self._food_seen = seen and not self._food_seen, seen
        if not fc:
            return {}
        hunger = self.drives.level(fc.get("drive", "food"))
        sr = fc.get("search")
        if not seen:
            if (sr and hunger >= float(sr.get("above", 0.5))
                    and self.t - self._searched_at >= int(sr.get("every", 8))):
                return {sr.get("verb", "look"): round(float(sr.get("gain", 0.3)) * hunger, 4)}
            return {}
        # the cue's strength: the `food_cue_gain` knob when the genome has one (ghrelin raises it, 10aa)
        pull = round(float(self.knobs.get("food_cue_gain", fc.get("gain", 0.5))) * hunger, 4)
        if onset and self.attention and float(fc.get("bid", 0.0)) > 0 and hunger >= float(fc.get("bid_above", 0.5)):
            self._offer("food in sight", float(fc["bid"]) * hunger)
        if pull <= 0:
            return {}
        out = {v: round(pull * float(fc["approach"]), 4) for v in far}
        out.update({v: pull for v in food_by_verb})
        return out

    def _hunger_recall(self, k: int) -> tuple[list, list]:
        """Hunger brings back the memories of eating (step 10ab): state-dependent recall, the
        need as a retrieval cue (Tulving & Thomson 1973; ghrelin acts on the hippocampus, Diano
        2006; hungry people's thoughts turn to food). The cue is what the mind learnt fed it
        (`food_words`), above loop.food_cue.recall_above. ([], []) otherwise."""
        fc = self.loop.get("food_cue") or {}
        above = fc.get("recall_above")
        if above is None or not self.food_words or self.drives.level(fc.get("drive", "food")) < float(above):
            return [], []
        cue = " ".join(self.food_words)
        return self.memory.recall(cue, self.t, k), self.memory.knowledge(cue, k)

    def _note_query(self, percepts, cue: str) -> str | None:
        """With the cue index, notes are asked for by what is near (cue-near); None otherwise."""
        ci = self.memory.cue_index
        return ci.query(percepts, cue) if ci is not None else None

    def _workspace(self, percepts, verbs) -> dict:
        cue = " ".join(percepts) + " " + (self.goal or "")
        k = self._recall_k("recall_for_qwen", 3)
        h_eps, h_notes = self._hunger_recall(1)          # hunger's memory comes first (10ab)
        eps = _dedupe(h_eps + self.memory.recall(cue, self.t, k), "t")[:k]
        notes = _dedupe(h_notes + self.memory.knowledge(cue, k, query=self._note_query(percepts, cue)),
                        "title")[:k]
        return {
            "t": self.t,
            "percepts": percepts,
            "recent": list(self.working),
            "memories": [f"(t={e['t']}) {e['text']}" for e in eps],
            "knowledge": [f"{n['title']}: {n['text']}" for n in notes],
            "_known": [n["title"] for n in notes],
            "open": list(self.open_contradictions),
            "expectations": (self.predictor.expectations(getattr(self, "_situation", " ".join(percepts)), verbs,
                                                         getattr(self, "_focus_by_verb", {}))
                             + self.predictor.social_expectations(others_here(percepts), verbs)),
            "last_expectation": dict(self.expectation) if self.expectation else None,
            "felt": self.drives.felt(),
            "felt_neutral": self.loop.get("felt_neutral", "calm and content"),
            **({"stuck": line} if (line := self._stuck_line()) else {}),
            "motive": self.drives.motive(),
            "goal": self.goal,
            **({"task": self.task} if self.loop.get("tasks") else {}),
            "verbs": verbs,
            "verb_counts": dict(self.verb_counts),
            "dials": self.dials.snapshot(),
            "temperature": self.knobs.get("qwen_temperature"),
        }

    def _remember(self, s: Stream, text: str, strength: float, **extra) -> None:
        # social memory (oxytocin, Rimmele 2009): a moment with someone in it is kept more strongly
        if "someone with" in text:
            strength += self.knobs.get("social_tag", 0.0)
        ep = self.memory.write({"t": self.t, "text": text, "strength": round(strength, 3), **extra})
        self._log.append({"kind": "episode", "id": ep["id"], "text": text, "strength": round(strength, 3),
                          **{("type" if k == "kind" else k): v for k, v in extra.items()}})
        self.lifetime["episodes"] += 1
        s.emit("brain", "memory_written", strength=strength)

    def _tag_recent(self) -> None:
        """The emotional tag (adrenaline, McGaugh 2004; Cahill 1994): an arousal surge
        strengthens the memory of what came just BEFORE it (post-training epinephrine
        enhances consolidation), so the episodes of the last `tag_window` ticks get the
        `emotional_tag` knob added to their strength, once (the highest surge counts). Strength
        decides which episodes sleep replays and how readily they are recalled."""
        tag = round(self.knobs.get("emotional_tag", 0.0), 3)
        if tag <= 0.0:
            return
        since = self.t - int(self.loop.get("tag_window", 10))
        if self._habit_store:                        # the same moments weigh more in tonight's habits
            self._habit_store.tag(since, self.t, tag)
        for ep in reversed(self.memory.episodes):
            if ep.get("t", -1) < since or ep.get("consolidated"):
                break
            if tag > ep.get("tag", 0.0):
                ep["strength"] = round(ep.get("strength", 0.0) - ep.get("tag", 0.0) + tag, 3)
                ep["tag"] = tag

    def _temperament(self, bundle, cfg: dict) -> dict | None:
        """Effectance's temperament: drawn once per individual (seeded by its name) from the
        genome's ranges, then written into its genome file (brain.json `temperament`), where
        step 11 could let it be inherited. None without `loop.effectance`."""
        eff = self.loop.get("effectance")
        if not eff:
            return None
        if cfg.get("temperament"):
            return cfg["temperament"]
        rng = random.Random(f"{bundle.path.name}:temperament")
        draw = lambda k, d: round(rng.uniform(*eff[k]), 4) if isinstance(eff.get(k), list) else float(eff.get(k, d))
        tm = {"birth": draw("birth", 1.0), "floor": draw("floor", 0.2), "half_life": draw("half_life", 5000)}
        path = bundle.path / "brain.json"
        g = json.loads(path.read_text(encoding="utf-8"))
        g["temperament"] = tm
        path.write_text(json.dumps(g, indent=2, ensure_ascii=False), encoding="utf-8")
        return tm

    def effect_appetite(self) -> float:
        """How much acting with effect pleases at this age: highest at birth, slowly lower
        with age (lifetime ticks), never below the individual's playful floor."""
        tm = self.temperament or {}
        b, f, h = tm.get("birth", 1.0), tm.get("floor", 0.2), max(1.0, tm.get("half_life", 5000))
        return f + (b - f) * 0.5 ** (self.lifetime.get("ticks", 0) / h)

    def _effect(self, s: Stream, what: str, frame: dict) -> None:
        """My act changed something: `effectance`, scaled by the age appetite and habituating
        to the same effect (the tenth door opening is not the first); drives and dials read it."""
        eff = self.loop.get("effectance")
        if not eff:
            return
        n = self.effect_counts.get(what, 0.0)
        amount = self.effect_appetite() * math.exp(-n / float(eff.get("habituation", 6.0)))
        self.effect_counts[what] = n + 1.0
        s.emit("brain", "effectance", amount=round(amount, 4))
        frame["effectance"] = round(amount, 4)

    def _fall_asleep(self, s: Stream) -> None:
        self.asleep = True
        self._prev, self._last_act = None, None          # the fast level chooses afresh at waking
        self.working.clear()                             # sleep wipes working memory
        self.task = None                                 # and today's task with it
        self.lifetime["sleeps"] += 1
        ic = self.loop.get("intentions") or {}
        if ic.get("through_sleep") and self.goal:        # an unfinished intention, kept for the night (3n Q1)
            self._remember(s, f"Before I slept, I still meant to: {self.goal}",
                           1.0 + float(ic.get("unfinished_tag", 0.0)), kind="intention")
        eff = self.loop.get("effectance")
        if eff and self.effect_counts:                   # habituation to effects wears off overnight
            keep = float(eff.get("recovery_per_sleep", 0.5))
            self.effect_counts = {k: round(v * keep, 3) for k, v in self.effect_counts.items() if v * keep >= 0.05}
        self._start_consolidation(s)
        if self.habits:
            self._sleep_habits(s)
        if self.head:
            # the day's choices rehearsed: the fast head is distilled from the middle level's (while the slow
            # level writes the notes); synchronous, ~10-20 s on the card
            rep = habits.distill_night(self.fast, self._mind, self._habit_store, self.head,
                                       seed=f"{self.bundle.path.name}:{self.t}")
            s.emit("brain", "head_night", accepted=1.0 if rep.get("accepted") else 0.0)
            self._head_night = rep

    def _sleep_habits(self, s: Stream) -> None:
        """The day is replayed and System 1's head learns (while System 2 writes the notes)."""
        if hasattr(self.fast, "train_head"):
            rep = habits.night(self.fast, self._mind, self._habit_store, self.habits,
                               seed=f"{self.bundle.path.name}:{self.t}")
        else:                                        # the stubs cannot learn: the day is let go
            self._habit_store.clear_day()
            rep = {"accepted": False, "why": "System 1 cannot learn"}
        s.emit("brain", "habits", accepted=1.0 if rep.get("accepted") else 0.0)
        self._habits_night = rep

    def _start_consolidation(self, s: Stream) -> None:
        eps = self.memory.unconsolidated(int(self.slow_cfg.get("consolidate_max_episodes", 60)))
        if eps and not self.consolidating:
            self._consolidating_upto = self.memory.episodes[-1]["id"]
            self.consolidating = True
            if self.slow_cfg.get("review_prompt") and self.memory.notes:
                # step 10b: the notes are hypotheses; the day tests them before it is consolidated
                self.slow.submit({"episodes": eps, "notes": self.memory.all_notes()}, self.t, kind="review")
                s.emit("brain", "review_started", notes=len(self.memory.notes))
            else:
                self._submit_consolidation(s, eps)

    def _submit_consolidation(self, s: Stream, eps: list | None = None) -> None:
        eps = eps or self.memory.unconsolidated(int(self.slow_cfg.get("consolidate_max_episodes", 60)))
        traces = [n["text"] for n in self.memory.provisional.values()]
        self.slow.submit({"episodes": eps, "notes": self.memory.all_notes(),
                          "open": list(self.open_contradictions),
                          **({"traces": traces} if traces else {})}, self.t, kind="consolidation")
        self._traces_today = 0
        s.emit("brain", "consolidation_started", episodes=len(eps))

    def _integrate_review(self, s: Stream, res: dict) -> dict:
        """Apply the verdicts, retire the notes that keep failing, then consolidate what
        survived (a failed review skips selection, never the consolidation)."""
        if res.get("error"):
            s.emit("slow", "slow_error", error=res["error"])
            out = {"error": res["error"]}
        else:
            verdicts = {}
            for v in res.get("verdicts", []):
                word = str(v.get("verdict", "")).strip().lower()
                verdicts[str(v.get("title", ""))] = ("held" if word.startswith("held") else
                                                    "failed" if word.startswith("fail") else "untested")
            out = self.memory.review(verdicts, self.t, window=int(self.loop.get("review_window", 6)),
                                     retire_after=int(self.loop.get("retire_after", 2)))
            self._log.append({"kind": "review", "verdicts": verdicts, "retired": list(out["retired"])})
            self.lifetime["notes_retired"] = self.lifetime.get("notes_retired", 0) + len(out["retired"])
            for _ in out["retired"]:
                s.emit("memory", "note_retired")
            out.update(seconds=res.get("seconds"), tokens=res.get("tokens"))
        self._submit_consolidation(s)
        return out

    def _integrate_consolidation(self, s: Stream, res: dict) -> dict:
        self.consolidating = False
        if res.get("error"):
            s.emit("slow", "slow_error", error=res["error"])
            return {"error": res["error"]}
        before = set(self.memory.notes)
        created, updated = self.memory.write_notes(res.get("notes", []), self.t)
        self._log.append({"kind": "notes", "notes": [
            {"title": str(n.get("title", "")).strip(), "text": str(n.get("text", "")).strip(),
             "new": slug(str(n.get("title", "")).strip()) not in before}
            for n in res.get("notes", []) if n.get("title") and n.get("text")],
            "contradictions": res.get("contradictions", []), "resolved": res.get("resolved", [])})
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

    def _start_cues(self, s: Stream) -> None:
        """After the consolidation, System 2 writes the cues of the notes this sleep touched. The mind
        does not wait for them (owner, 10ae): it wakes when the consolidation is done, as without cues,
        and the cues go into the wiki (and the index is rebuilt from it) when they land, a few ticks
        into the morning. A sleep that starts before they land asks for none: its notes wait for the
        next night. Nothing to cue, or a call in flight: the index is rebuilt at once (the review may
        have retired notes)."""
        if self.memory.cue_index is None:
            return
        notes = self.memory.needs_cues()
        if not notes or self._cues_job is not None:
            if self._cues_job is not None:
                s.emit("brain", "cues_skipped", notes=len(notes))
            self.memory.rebuild_index()
            return
        self._cues_job = (self.t, [n["title"] for n in notes])
        top = (self.loop.get("recall_index") or {}).get("acts_did_top")
        self.slow.submit({"notes": [{"title": n["title"], "text": n["text"]} for n in notes],
                          "verbs": dict(self.verbs_seen),
                          **({"did": self._acts_did(int(top))} if top else {})}, self.t, kind="cues")
        s.emit("brain", "cues_started", notes=len(notes))

    def _acts_did(self, top: int) -> list[str]:
        """What each act did, from the mind's own episodes as they are at this sleep ("... I chose to X.
        <outcome>"): the action-outcome links the goal-directed system learns (Balleine & Dickinson 1998),
        read by the cue writing so it can name the act by which the mind did what a line says (v4). Per
        act, its `top` most frequent outcomes (loop.recall_index.acts_did_top), with how often."""
        seen: dict[str, Counter] = {}
        for e in self.memory.episodes:
            m = _CHOSE.search(e.get("text", ""))
            if m:
                seen.setdefault(m.group(1), Counter())[m.group(2).strip()] += 1
        out = []
        for v in sorted(seen):
            out += [f"- {v}: {o}" + (f" ({n} times)" if n > 1 else "") for o, n in seen[v].most_common(top)]
        return out

    def _integrate_cues(self, s: Stream, res: dict) -> dict:
        asked_t, asked = self._cues_job or (self.t, [])
        self._cues_job = None
        # when it landed: ticks since it was asked, and since the mind woke (None: still asleep)
        late = {"ticks": self.t - asked_t,
                "after_wake": None if self.asleep or self._woke_at is None else self.t - self._woke_at}
        if res.get("error"):
            s.emit("slow", "slow_error", error=res["error"])
            out = {"error": res["error"], "asked": len(asked)}      # asked again at the next sleep
        else:
            # stamped with the tick they were asked for: a note revised since (a sleep that started
            # before they landed) keeps needing its cues
            v4 = bool((self.loop.get("recall_index") or {}).get("acts_did_top"))
            out = self.memory.write_cues(asked, res.get("cues", []), asked_t,
                                         acts=self.verbs_seen if v4 else None)
            out["written"] = [{"title": c.get("title"), "when": c.get("when"), "recall": c.get("recall"),
                               **({"act": c.get("act")} if v4 else {})}
                              for c in res.get("cues", [])][:40]
        out.update(units=self.memory.rebuild_index(), habituated=self.memory.cue_index.habituated,
                   seconds=res.get("seconds"), tokens=res.get("tokens"), **late)
        return out

    # -- attention (step 10d) ------------------------------------------------
    def _weight(self, source: str) -> float:
        return float(self.attention.get("weights", {}).get(source, 0.0))

    def _fade(self, priority: float, since: int) -> float:
        """Bids and a thought's protection halve every `half_life` ticks: both are about
        a moment that is slipping away."""
        return priority * 0.5 ** ((self.t - since) / float(self.attention.get("half_life", 10)))

    def _resting(self, reason: str) -> bool:
        """Right after a thought lands, the chronic sources (the genome's `rest_blocks`:
        System 1's doubt, a need that stays unmet) are not heard for `rest_after_thought`
        ticks (a knob): the body carries out the intention before the mind thinks again.
        A surprise, an alarm, being spoken to or a spent intention still are."""
        return (self.t - self._landed_at < self.knobs.get("rest_after_thought", 0)
                and reason.split(":")[0] in self.attention.get("rest_blocks", []))

    def _offer(self, reason: str, priority: float) -> None:
        """A bid for System 2; only the strongest waiting one is kept."""
        if self._resting(reason):
            return
        if priority > 0 and priority > (self._fade(self._bid[1], self._bid[2]) if self._bid else 0.0):
            self._bid = (reason, round(priority, 4), self.t)

    def _frozen(self) -> list[str] | None:
        """The verbs left to System 1 while an orienting thought is in flight, or None."""
        still = self.attention.get("freeze_verbs") if self.attention else None
        if not still or not self._thought_job or self._thought_job[1] != "surprise":
            return None
        if self.t - self._thought_job[3] > int(self.attention.get("freeze_max", 30)):
            return None                                      # never held still for long
        return still

    def _attend(self, s: Stream, percepts, verbs, conf: float, motive, need: float, frame: dict):
        """The salience network: collect this tick's bids, then give System 2 the strongest
        if it is free, or let it interrupt the thought in flight if it beats that thought's
        (fading) priority by the `interrupt_margin` knob. Returns the reason or False."""
        worst = {n for n, top in self.top_band.items() if self.drives.level(n) >= top}
        if motive and self.t - self._last_need_bid >= self.loop.get("need_escalation_every", 10):
            if motive in worst and motive not in self._in_worst:
                # entering the worst band is an alarm; staying in it is not (habituation)
                self._offer(f"worst band: {motive}", self._weight("worst_band"))
                self._last_need_bid = self.t
            elif need >= self.loop.get("escalate_if_need_above", 1.1):
                stuck = self.stuck is not None and motive == self.stuck.get("drive", "frustration")
                if stuck and self._stuck_line():
                    self._offer(f"stuck: {self._stuck_line()}", self._weight("stuck") * need)
                else:
                    self._offer(f"unmet need: {motive}", self._weight("need") * need)
                self._last_need_bid = self.t
        self._in_worst = worst
        if conf is not None and conf < self.knobs.get("escalate_below", 0.0):
            # a split vote (the middle level's, when it was asked): the anterior cingulate's conflict signal
            self._offer("low confidence", self._weight("low_confidence") * (1.0 - conf))
        if self._bid and self._resting(self._bid[0]):
            self._bid = None                                 # an older chronic bid: not heard now
        self._free_ticks = 0 if (self._thought_job or self.slow.busy) else self._free_ticks + 1
        if (not self._bid and self._weight("idle") > 0
                and self._free_ticks > int(self.attention.get("idle_after", 3))):
            # the default mode network (step 10ad): nothing asks for System 2 and it has been free
            # for a moment, so it thinks unasked, as minds wander when the task leaves them free
            # (Raichle 2001; Killingsworth & Gilbert 2010); the weakest source, it yields to any other
            self._offer("idle", self._weight("idle"))
        if self.t - self._landed_at < self.knobs.get("rest_after_thought", 0):
            frame["resting"] = self.t - self._landed_at
        if not self._bid:
            return False
        reason, _, since = self._bid
        value = self._fade(self._bid[1], since)
        if value < float(self.attention.get("bid_floor", 0.05)):
            self._bid = None
            return False
        if self.drives.urgency("hunger") >= self.loop.get("think_if_hunger_below", 1.0):
            return False                                     # too drained to think: the bid waits
        level = self._fade(self._thought_job[2], self._thought_job[3]) if self._thought_job else None
        margin = self.knobs.get("interrupt_margin", 0.1)
        source = reason.split(":")[0]
        # some sources only wait for a free System 2 (the genome's `cannot_interrupt`: System 1's
        # chronic doubt), and a source never interrupts its own thought (a fresher version of the
        # same concern is not a new alarm)
        blocked = level is not None and (source in self.attention.get("cannot_interrupt", [])
                                         or source == self._thought_job[1].split(":")[0])
        if level is not None and (blocked or value <= level + margin):
            frame["attention"] = {"waiting": reason, "bid": round(value, 3),
                                  "thought": self._thought_job[1], "level": round(level, 3)}
            return False
        if level is not None:
            s.emit("brain", "interrupted", level=level)
            frame["attention"] = {"interrupted": self._thought_job[1], "level": round(level, 3),
                                  "by": reason, "bid": round(value, 3)}
        self._job_seq += 1
        ws = dict(self._workspace(percepts, verbs), _job=self._job_seq)
        if self.memory.cue_index is not None:        # the notes System 2 was given (the cue arm's gate)
            frame["ws_known"] = ws["_known"]
        if reason == "to speak":
            ws["urge"] = "I feel the urge to say something aloud."
        if reason == "surprise" and self.attention.get("surprise_think_budget"):
            ws["_think_budget"] = int(self.attention["surprise_think_budget"])
        effort = self._effort(value)
        if effort:
            ws["_effort"] = effort
            frame["effort"] = effort
        if reason == "idle" and self.attention.get("idle_think_budget"):
            ws["_think_budget"] = int(self.attention["idle_think_budget"])   # to land inside the rest
        self.slow.submit(ws, self.t, kind="idle" if reason == "idle" else "thought")
        self._log_asked(ws, reason)
        late = int((self.loop.get("intentions") or {}).get("late_within", 0))
        self._asked = {j: a for j, a in self._asked.items() if self.t - a[0] <= late}
        self._asked[self._job_seq] = (self.t, self.drives.motive())
        self._thought_job = (self._job_seq, reason, value, self.t)
        self._bid = None
        self._addressed_at = self._surprised_at = None
        s.emit("brain", "escalated", reason=reason, confidence=conf)
        self.last_escalated = self.t
        return reason

    def _recall_k(self, key: str, default: int) -> int:
        """How many memories come back: the loop's number, scaled by the `recall_scale` knob
        (stress hormones impair retrieval, de Quervain 1998); rounded half up."""
        return int(math.floor(int(self.loop.get(key, default)) * self.knobs.get("recall_scale", 1.0) + 0.5))

    # -- logs for the inspector (PLAN 1.5, P1 step 5) -------------------------
    def _effort(self, stakes: float) -> str | None:
        """The effort a thought gets (3n I6; genome `knobs.think_effort` and `slow.effort`): the knob (noradrenaline's
        inverted U, compute hunger, a pressing need) plus `stakes_gain` x the bid's value, read against the bands
        (none / low / medium). Mental effort is spent by its expected value against its cost (Shenhav, Botvinick &
        Cohen 2013; Kurzban et al. 2013). None when the genome has no bands: the slow level's own setting."""
        cfg = self.slow_cfg.get("effort") or {}
        if not cfg.get("bands") or "think_effort" not in self.knobs:
            return None
        x = float(self.knobs["think_effort"]) + float(cfg.get("stakes_gain", 0.0)) * stakes
        names = [name for _, name in cfg["bands"]]
        i = next((k for k, (upto, _) in enumerate(cfg["bands"]) if x < float(upto)), len(names) - 1)
        # an effort kept for the dark (3n Q2): in the light, the band below
        dark = bool(self._entrain.get("dark")) if self.entrain else False
        while i > 0 and names[i] in cfg.get("dark_only", []) and not dark:
            i -= 1
        return names[i]

    def _log_asked(self, ws: dict, reason: str) -> None:
        """A thought asked of the slow level: its job, why, and the text it was given."""
        render = getattr(self.slow, "render", None)
        self._log.append({"kind": "thought_asked", "job": ws.get("_job"), "reason": reason,
                          "prompt": render(ws) if render else " ".join(ws.get("percepts", []))})

    def declarations(self, verbs: dict | None = None) -> dict:
        """What the genome declares, for the run's meta row: the inspector draws whatever is here (a new drive,
        dial or knob appears without touching the page)."""
        def clean(d):
            return {k: v for k, v in d.items() if not str(k).startswith("_")}
        return {
            "drives": {d.name: {"enabled": d.enabled, "setpoint": d.setpoint, "felt": d.felt}
                       for d in self.drives.drives},
            "dials": {k: clean(v) for k, v in self.bundle.dials.items() if not k.startswith("_") and isinstance(v, dict)},
            "knobs": {k: clean(v) for k, v in self.knobs_cfg.items() if not k.startswith("_") and isinstance(v, dict)},
            "verbs": {**(verbs or {}), **self.internal_verbs},
            "levels": {"fast": getattr(self.fast, "name", None), "middle": getattr(self.middle, "name", None),
                       "slow": getattr(self.slow, "name", None)},
            "gate": clean(self.gate.get("signals", {})) if self.gate else None,
            "head": self.head["mode"] if self.head else None,
            "attention": clean(self.attention.get("weights", {})) if self.attention else None,
        }

    # -- the fast level (three levels, PLAN 1.1) ----------------------------
    def _bands(self) -> dict[str, int]:
        """How many of its felt bands each drive is in."""
        return {d.name: sum(1 for th, _ in d.felt if d.level >= th) for d in self.drives.active if d.felt}

    def _gate_signals(self, sensed, percepts, addressed: bool, heard: bool) -> dict[str, float]:
        """What calls for a new choice this tick, each signal at its genome weight (`fast.gate.signals`), so the
        tick's salience is the strongest (the thalamus and the salience network: a change, a failure, a
        surprise, an alarm catch attention; a routine does not). All generic, nothing of the world:
          woke       the first awake tick (a start, a waking): choose afresh
          front      what is in front or held changed (the world's `focus`, else the whole situation)
          others     who is in sight changed
          alarm      a drive crossed into a higher felt band (a crossing, not a level: it fires once)
          intention  the slow level gave a new intention or task
          effect     my last act changed the world or what I hold
          fail       my last act changed nothing, the first time in a row (not an act meant to change nothing:
                     `fast.gate.fail_ignores`; the world says whether it changed anything, else the situation does)
          novel      my last act's outcome was never met before
          surprise   the cerebellum's surprise at my last act (times its size)
          addressed  spoken to; heard: words or a call heard
          long       nothing flagged for `fast.gate.long_after` ticks (the routine is checked now and then)
          need       a drive's urgency at or above `need_above`, nothing flagged for `need_every` ticks (times the
                     urgency): a pressing need keeps returning to attention, not only when it crosses a band
          relief     what relieves a felt need came into view, in front or beside (times the need's urgency)"""
        w = self.gate.get("signals", {})
        focus = next((e.data["focus"] for e in sensed if e.data.get("focus")), None) or " ".join(percepts)
        now = {"focus": focus, "others": sorted(others_here(percepts)), "bands": self._bands(),
               "goal": (self.goal, self.task), "percepts": list(percepts)}
        prev, self._prev = self._prev, now
        fired: dict[str, float] = {}

        def fire(name: str, x: float = 1.0) -> None:
            if float(w.get(name, 0.0)) > 0 and x > 0:
                fired[name] = round(float(w[name]) * x, 4)

        if prev is None:
            fire("woke")
        else:
            if focus != prev["focus"]:
                fire("front")
            if now["others"] != prev["others"]:
                fire("others")
            if any(n > prev["bands"].get(k, n) for k, n in now["bands"].items()):
                fire("alarm")
            if now["goal"] != prev["goal"] and any(now["goal"]):
                fire("intention")
            la = self._last_act
            if la:
                if la["effect"]:
                    fire("effect")
                changed = la["changed"] if la["changed"] is not None else percepts != prev["percepts"]
                if (not changed and la["streak"] <= 1 and la["act"] not in INTERNAL
                        and la["act"] not in self.gate.get("fail_ignores", [])):
                    fire("fail")
                if la["novel"] >= 1.0:
                    fire("novel")
        fire("surprise", self._gate_surprise)
        fire("relief", self._relief_onset)               # what relieves a felt need came into view (3n I5)
        if addressed:
            fire("addressed")
        if heard:
            fire("heard")
        if self.t - self._last_flag >= int(self.gate.get("long_after", 10**9)):
            fire("long")
        # a need re-flags, except those the genome leaves to a reflex (`need_ignores`, 3k: tiredness flips the sleep
        # switch, doze, rather than calling a deliberation; isl8: it sent 95 of 181 offered night ticks to the middle
        # level, which took sleep on 3)
        ignored = set(self.gate.get("need_ignores", []))
        need = max((u for n, u in self.drives.urgencies().items() if n not in ignored), default=0.0)
        above = float(self.gate.get("need_above", 1.1))
        every = float(self.gate.get("need_every", 10**9))
        if need >= above and "need_every_min" in self.gate and above < 1.0:
            lo = float(self.gate["need_every_min"])     # the more pressing, the sooner it returns
            every -= (every - lo) * min(1.0, (need - above) / (1.0 - above))
        if need >= above and self.t - self._last_flag >= round(every):
            fire("need", need)                           # a pressing need keeps coming back (interoception)
        return fired

    def _carry_on(self, verbs: dict) -> tuple[str, str]:
        """Nothing calls for a new choice: carry on (`fast.carry`). Repeat the last act while it changes
        something (walking on), else follow the intention's act, else the idle act (the first of
        `fast.carry.idle` offered): routines run without the supervisor (Norman & Shallice). Instead of
        idling, a sleepy body dozes off (the `doze` knob, when sleep is offered), a restless body searches (the `restless` knob, the chance a tick): it goes on one of
        `fast.carry.search`'s verbs, the same one as before unless that one just changed nothing."""
        la = self._last_act
        if (self.carry.get("repeat", True) and la and la["changed"] and la["act"] in verbs
                and la["act"] not in INTERNAL):
            return la["act"], "repeat"
        # a frustrated body stops following the intention's act into what it just failed at (3n I2; genome
        # `fast.carry.stuck_above`, the frustration level from which it lets go)
        let_go = (self.stuck is not None and la and la["act"] == self.goal_verb and la["changed"] is False
                  and self.drives.level(self.stuck.get("drive", "frustration"))
                  >= float(self.carry.get("stuck_above", 1.1)))
        if (self.carry.get("follow", True) and self.goal_verb in verbs and self.goal_verb not in INTERNAL
                and not let_go):
            return self.goal_verb, "intention"
        # with nothing to do and sleep offered (the pressure above the gate), the body may drift off: the
        # `doze` knob, the chance a tick (3k, isl7: at night 126 of 175 offered ticks were idle `wait`s, the
        # middle level never asked); falling asleep is the sleep switch flipping, not a decision (Saper 2001)
        if "sleep" in verbs and self._rng.random() < float(self.knobs.get("doze", 0.0)):
            return "sleep", "doze"
        search = [v for v in self.carry.get("search", []) if v in verbs]
        if search and self._rng.random() < float(self.knobs.get("restless", 0.0)):
            way = self._search_way
            if way not in search or (la and la["act"] == way and la["changed"] is False):
                way = self._rng.choice([v for v in search if v != way] or search)
            self._search_way = way
            return way, "search"
        idle = [v for v in self.carry.get("idle", ["wait", "rest"]) if v in verbs]
        return (idle[0] if idle else ("rest" if "rest" in verbs else next(iter(verbs)))), "idle"

    def _strong_moment(self, s: Stream, before: dict) -> None:
        """Learning within the day (genome `traces`; docs, Consolidation, section 06): when this tick's act
        relieved a need at once (its level fell by `relief_above` beyond the drift) or made one worse (rose
        by `harm_above`), a provisional trace goes into the cue index now: `when` = what was near, `recall`
        = the world's own words and what the body felt (the drive's `relieved` / `worsened` phrase), `act` =
        the act for a relief, none for a harm. A relief that used something up credits the last other act
        before it that changed what I hold or the world, within `chain_ticks` (the eligibility trace of
        dopamine: taking the food, then eating it): the trace is then that chain's, at the source act's place,
        and the relieving act has none of its own (3k). The night's consolidation sees them; the rebuild of
        the index after it drops them. One-trial learning (Garcia; the hippocampus's fast encoding,
        McClelland 1995), sorted by sleep. Nothing of the world: needs, acts and the world's words."""
        tr, cur = self.traces, self._trace_hist[-1]
        if self._traces_today >= int(tr.get("max_per_day", 12)) or not cur["outcome"]:
            return
        for d in self.drives.active:
            drift = d.drift_asleep if self.asleep and d.drift_asleep is not None else d.drift
            delta = d.level - before.get(d.name, d.level) - drift
            if d.name in tr.get("relieved", {}) and delta <= -float(tr.get("relief_above", 0.08)):
                felt = tr["relieved"][d.name]
                # with a source, only the chain's trace, keyed where the source act was (3k, isl7: the act's own
                # trace was keyed on what lay in front while eating, rocks, the sea, and came back there); a
                # source only when the relief itself used something up (eating what I hold, not a drink at a
                # spring: that relief belongs to the place, whatever was taken before it)
                src = next((h for h in reversed(list(self._trace_hist)[:-1])
                            if h["effect"] and h["outcome"] and h["act"] != cur["act"]), None) if cur["effect"] else None
                done = ([(src["near"], f"Here: {src['outcome']} Then: {cur['outcome']} {felt}", src["act"])] if src
                        else [(cur["near"], f"Here: {cur['outcome']} {felt}", cur["act"])])
            elif d.name in tr.get("worsened", {}) and delta >= float(tr.get("harm_above", 0.05)):
                felt = tr["worsened"][d.name]
                done = [(cur["near"], f"Here: {cur['outcome']} {felt}", None)]
            else:
                continue
            for when, recall, act in done:
                key = f"trace-{self.t}-{len(self.memory.provisional)}"
                if self.memory.add_trace(key, when, recall, act, self.t):
                    self._traces_today += 1
                    s.emit("memory", "trace_written", drive=d.name, delta=round(delta, 4))
                    self._log.append({"kind": "trace", "t": self.t, "drive": d.name, "delta": round(delta, 4),
                                      "when": when, "recall": recall, "act": act})

    def _entrain_step(self, summary: dict, before: dict) -> None:
        """Entrainment (genome `entrain`, 3k; owner: "le calage devrait être détecté par le brain"). The tick's
        worlds are not the body's: a day may be 180 ticks or 110, and a sleep pressure built for one collapses
        a body by the afternoon of the other (isl7: full in ~100 of ~126 waking ticks). So the body measures its
        day: the run of light between two darks (the world's `dark`, which melatonin reads; a world without it
        never entrains) and the drive's rise a waking tick at pace 1; at nightfall it moves the drive's pace
        (`adapt` of the way) toward the one that would have brought it to `target` then. Modelled loosely on
        the infant's sleep, polyphasic at birth and gathered into the night over weeks as the clock entrains
        (a design of ours: in adults it is the clock that follows the photoperiod, Wehr 1991, not the
        homeostat's rate; here the tick is arbitrary, so the rate has to be set against the day). The pace
        starts from the genome's `init` (the island's: what a species of that island would inherit): replayed
        on isl7, it holds from a start near right and does not find its way within 6 days from 1.0."""
        en, st = self.entrain, self._entrain
        name = en.get("drive", "sleep_pressure")
        dark = summary.get(en.get("signal", "dark.amount"), 0.0) >= float(en.get("dark_above", 0.5))
        if dark and not st["dark"] and st["light"] >= int(en.get("min_light", 30)) and st["n"]:
            # the day's measures, smoothed over days (`smooth`: the thoughts' rate varies from day to day, and a
            # pace set on one busy day collapsed the next in the replay)
            sm = float(en.get("smooth", 0.3))
            st["raw_m"] = st["rise"] / st["n"] if st.get("raw_m") is None else                 st["raw_m"] + sm * (st["rise"] / st["n"] - st["raw_m"])
            st["light_m"] = st["light"] if st.get("light_m") is None else                 st["light_m"] + sm * (st["light"] - st["light_m"])
            raw = st["raw_m"]
            if raw > 0:
                want = float(en.get("target", 0.75)) / (raw * st["light_m"])
                old = st["pace"]
                st["pace"] = clamp(old + float(en.get("adapt", 0.5)) * (want - old),
                                   float(en.get("min", 0.3)), float(en.get("max", 2.0)))
                st["days"] += 1
                self.drives.set_pace(name, st["pace"])
                self._log.append({"kind": "entrain", "t": self.t, "drive": name, "light": st["light"],
                                  "rise": round(raw, 5), "want": round(want, 3), "pace": round(st["pace"], 3)})
        if dark:
            st.update(light=0, rise=0.0, n=0)
        else:
            st["light"] += 1
            if not self.asleep and name in before and st["pace"] > 0:
                st["rise"] += (self.drives.level(name) - before[name]) / st["pace"]
                st["n"] += 1
        st["dark"] = dark

    def _note_act(self, action: str, events, novelty: float) -> None:
        """What the gate needs of the act just done: did it change anything (the world's `changed` on the
        outcome; None when the world does not say), did it have an effect, how new was its outcome, and how
        many times in a row this same act has changed nothing."""
        said = [e.data["changed"] for e in events if e.kind == "outcome" and "changed" in e.data]
        effect = any(e.kind == "effect" for e in events)
        changed = True if effect else (any(said) if said else None)
        if action in INTERNAL:
            changed = False
        la = self._last_act
        streak = 0
        if changed is False:
            streak = la["streak"] + 1 if la and la["act"] == action and la["changed"] is False else 1
        self._last_act = {"act": action, "changed": changed, "effect": effect, "novel": novelty,
                          "streak": streak}

    def _stuck_step(self, s: Stream, action: str, outcome_texts: list[str]) -> None:
        """Frustration's events (3n I2; genome `stuck`, the drive in `drives.json`): an act that changed nothing
        (the world's `changed`, as the `fail` signal reads it; not an act meant to change nothing) while something
        is wanted (a goal, a task, or a need at `stuck.active_above`) is `thwarted`, `stuck.same` times more when it
        is the same act meeting the same thing within `stuck.same_within` ticks; an act with an effect is `gained`,
        one that changed something else `went`. In people a blocked goal raises arousal and vigour (Amsel's
        frustration effect), varies behaviour and draws control to the conflict (Botvinick et al. 2001); what the
        mind then does is not coded."""
        la = self._last_act or {}
        st = self.stuck
        if la.get("changed") is False and action not in INTERNAL and action not in self.gate.get("fail_ignores", []):
            drive = st.get("drive", "frustration")
            wanted = bool(self.goal or self.task) or any(
                u >= float(st.get("active_above", 0.5)) for n, u in self.drives.urgencies().items() if n != drive)
            if not wanted:
                return
            met = self._focus_by_verb.get(action, getattr(self, "_situation", ""))
            last = self._stuck_last
            same = bool(last and last[0] == action and last[1] == met
                        and self.t - last[2] <= int(st.get("same_within", 10)))
            s.emit("brain", "thwarted", amount=float(st.get("same", 1.5)) if same else 1.0)
            self._stuck_last = (action, met, self.t)
            run = self._stuck_run
            run["n"] = run["n"] + 1 if run["act"] == action else 1
            run["act"], run["text"] = action, " ".join(outcome_texts)[:160]
        elif la.get("effect"):
            s.emit("brain", "gained")
            self._stuck_run = {"act": None, "n": 0, "text": ""}
        elif la.get("changed"):
            s.emit("brain", "went")

    def _stuck_line(self) -> str | None:
        """What keeps failing, for the slow level while frustration is felt: the mind's own recent history,
        counted (the working memory lists the outcomes, not how often the same one came back)."""
        if self.stuck is None:
            return None
        drive = next((d for d in self.drives.active if d.name == self.stuck.get("drive", "frustration")), None)
        run = self._stuck_run
        if not drive or not drive.felt or drive.level < drive.felt[0][0] or run["n"] < 2:
            return None
        return f"I have tried to {run['act']} {run['n']} times; {run['text']}"

    def _need_cues(self, sensed) -> None:
        """What relieves a felt need comes into view (3n I5; genome `loop.need_cues`: for each, the world's
        `<by>` on the percept, the drive it serves, the level from which it counts): the onset flags the gate at
        the drive's urgency (`fast.gate.signals.relief`) and may bid for the slow level (`bid`). An innate orienting
        to what the body lacks: thirst makes water cues wanted (Berridge 2004), as hunger does food (`food_cue`)."""
        self._relief_onset = 0.0
        for cue in self.loop.get("need_cues", []):
            by = next((e.data[cue["by"]] for e in sensed if cue["by"] in e.data), {})
            seen = frozenset(by.items())
            new = bool(seen - self._relief_seen.get(cue["by"], frozenset()))
            self._relief_seen[cue["by"]] = seen
            level = self.drives.level(cue["drive"])
            if not new or level < float(cue.get("above", 0.5)):
                continue
            u = self.drives.urgency(cue["drive"])
            self._relief_onset = max(self._relief_onset, u)
            if self.attention and float(cue.get("bid", 0.0)) > 0:
                self._offer(f"{cue['drive']} in view", float(cue["bid"]) * u)

    def _head_propose(self, state: dict, verbs: dict) -> dict | None:
        """The fast head's answer to the choice question (this mind's own head): its top act and that act's
        probability as its confidence."""
        try:
            p = self.fast.appraise(state, verbs, **self._mind_kw)
        except Exception as e:  # noqa: BLE001 - the head is an extra; the middle level still answers
            print(f"[head] {type(e).__name__}: {str(e)[:120]}", flush=True)
            return None
        probs = p.get("probabilities") or {}
        act = max(probs, key=probs.get) if probs else p.get("action")
        if act not in verbs:
            return None
        return {"action": act, "confidence": float(probs.get(act, p.get("confidence") or 0.0)),
                "probabilities": probs}

    def _head_decides(self, prop: dict) -> bool:
        """`fast.head.mode` decide: the head acts alone when it is sure (`decide_above`) and has agreed with the
        middle level often enough lately (`trust_above` over a full `trust_window`); a share (`audit`) of those
        ticks is still asked, so its record goes on."""
        h = self.head
        if h["mode"] != "decide" or prop["confidence"] < float(h["decide_above"]):
            return False
        w = self._head_trust
        if len(w) < w.maxlen or sum(w) / len(w) < float(h["trust_above"]):
            return False
        return self._rng.random() >= float(h["audit"])

    def _head_label(self, state: dict, verbs: dict, got: dict, prop: dict | None, info: dict) -> None:
        """The middle level's choice becomes tonight's label; the head's proposal, if any, is scored against it."""
        top = got.get("top", got["action"])
        probs = got.get("probabilities") or {got["action"]: 1.0}
        row = {"t": self.t, "state": {k: v for k, v in state.items() if not k.startswith("_")},
               "verbs": verbs, "probs": probs, "top": top}
        if prop:
            row["head"] = prop["action"]
            agreed = prop["action"] == top
            self._head_trust.append(1 if agreed else 0)
            info["head"]["agreed"] = agreed
        self._habit_store.add(row)

    def _fast_level(self, s: Stream, state: dict, verbs: dict, sensed, percepts, addressed: bool, heard: bool,
                    s2_act: str | None, frame: dict) -> dict:
        """The fast level: does this moment call for a new choice? If the slow level just chose, its act;
        if the gate flags the tick (its salience at or above the `gate_above` knob, else `fast.gate.above`),
        the middle level chooses; otherwise the body carries on. Logged per tick in `frame["fast"]`: the flag,
        the signals that fired, who chose, and the middle level's call."""
        fired = self._gate_signals(sensed, percepts, addressed, heard)
        salience = max(fired.values(), default=0.0)
        above = float(self.knobs.get("gate_above", self.gate.get("above", 0.3)))
        flag = salience > 0 and salience >= above
        info: dict = {"flag": flag, "why": fired}
        ap = None
        if s2_act:
            ap, info["by"] = {"action": s2_act, "confidence": None}, "slow"
        elif flag:
            prop = self._head_propose(state, verbs) if self.head else None
            if prop:
                info["head"] = {"act": prop["action"], "conf": round(prop["confidence"], 3)}
            if prop and self._head_decides(prop):
                ap, info["by"] = prop, "head"
            else:
                got = self.appraiser.appraise(state, verbs)
                if "seconds" in got:
                    info["seconds"] = got["seconds"]
                    s.emit("middle", "middle_call", tokens=got.get("tokens", 0), seconds=got["seconds"])
                if got.get("action") in verbs:
                    ap, info["by"] = got, "middle"
                    if "mass" in got:
                        info["mass"] = got["mass"]
                    if self.head:
                        self._head_label(state, verbs, got, prop, info)
                else:
                    info["error"] = got.get("error", f"no act: {got.get('action')!r}")
                    s.emit("middle", "middle_error")
        if ap is None:
            act, how = self._carry_on(verbs)
            ap, info["by"], info["carry"] = {"action": act, "confidence": None}, "carry", how
        if flag:
            self._last_flag = self.t
            s.emit("fast", "gate", salience=round(salience, 4))
        frame["fast"] = info
        return dict(ap, salience=salience)

    # -- the tick ---------------------------------------------------------
    def tick(self, world) -> dict:
        s = Stream(self.t)
        frame = {"t": self.t}
        self._log = []                                       # what changed this tick, in full text (the inspector)
        # hormones the body feels: the genome's `dials_as_events` become events, which drives
        # can rise with (cortisol and appetite); last tick's level, as a blood level lags
        levels = self.dials.levels()
        for name in self.loop.get("dials_as_events", []):
            if name in levels:
                s.emit("dial", name, level=round(levels[name], 4))

        # A goal held without being refreshed fades (prefrontal maintenance is costly).
        # Asleep, an intention is not maintained but stored (3n Q1, loop.intentions.through_sleep): weighed at waking.
        held_asleep = self.asleep and (self.loop.get("intentions") or {}).get("through_sleep")
        if self.goal and not held_asleep and self.t - self.goal_set_at >= int(self.loop.get("goal_ttl", 30)):
            self.goal = self.goal_verb = None
            s.emit("brain", "goal_faded")
        if self.task and self.t - self.task_set_at >= int(self.loop.get("task_ttl", 10**9)):
            self.task = None
            s.emit("brain", "task_faded")

        # Finished System 2 jobs. A failed one still costs its tokens.
        thought = consolidation = review = cues = None
        for res in self.slow.poll(self.t):
            if res.get("kind") == "review":
                s.emit("slow", "dream_cost", tokens=res.get("tokens", 0), seconds=res.get("seconds", 0.0))
                review = self._integrate_review(s, res)
                continue
            if res.get("kind") == "consolidation":
                # dreaming is logged, not charged to hunger: sleep restores, it does not exhaust
                s.emit("slow", "dream_cost", tokens=res.get("tokens", 0), seconds=res.get("seconds", 0.0))
                consolidation = self._integrate_consolidation(s, res)
                self._start_cues(s)
                continue
            if res.get("kind") == "cues":
                s.emit("slow", "dream_cost", tokens=res.get("tokens", 0), seconds=res.get("seconds", 0.0))
                cues = self._integrate_cues(s, res)
                continue
            s.emit("slow", "qwen_call", tokens=res.get("tokens", 0), seconds=res.get("seconds", 0.0))
            if res.get("kind") == "idle":
                # a daydream is remembered, and steers nothing: no goal, pull, task or rest
                # (mind-wandering is decoupled from the task at hand; Smallwood & Schooler 2006).
                # One the task pulled attention away from still finishes and is remembered.
                if self._thought_job and res.get("job") == self._thought_job[0]:
                    self._thought_job = None
                else:
                    res["cut"] = True
                thought = thought or res
                if res.get("error") or not res.get("thought"):
                    s.emit("slow", "slow_error", error=res.get("error", "empty thought"))
                    continue
                s.emit("slow", "daydream")
                self.lifetime["daydreams"] = self.lifetime.get("daydreams", 0) + 1
                self._remember(s, f"I thought: {res['thought']}", 1.0, kind="thought")
                continue
            asked = self._asked.pop(res.get("job"), None)
            if self.attention:
                if not self._thought_job or res.get("job") != self._thought_job[0]:
                    if not self._lands_late(res, asked):
                        frame["superseded"] = {"goal": res.get("goal"), "job": res.get("job")}
                        self._log.append({"kind": "thought_cut", **{k: v for k, v in res.items() if k != "kind"}})
                        continue                             # interrupted: about a moment that is gone
                    # interrupted, but about a moment still here: it lands; the newer thought stays in flight
                    frame["late"] = {"job": res.get("job"), "asked": asked[0]}
                    self._log.append({"kind": "thought_late", **{k: v for k, v in res.items() if k != "kind"}})
                else:
                    self._thought_job = None
            thought = res
            if res.get("error") or not res.get("goal"):
                s.emit("slow", "slow_error", error=res.get("error", "empty goal"))
                continue
            self.goal, self.goal_verb = res["goal"], res.get("action")
            self.goal_motive = asked[1] if asked else self.drives.motive()
            self._last_follow = None
            self.goal_set_at = self._landed_at = self.t
            if res.get("task_done") is True and self.task:
                s.emit("slow", "task_done")
                frame["task_done"] = self.task
                self.task = None
            if res.get("task") and res["task"] != self.task:
                self.task, self.task_set_at = res["task"], self.t
                s.emit("slow", "task_set")
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

        # the wake threshold: a knob when the genome declares one (melatonin holds sleep through the night:
        # Borbély's two processes, the pressure falling toward a threshold the circadian signal moves)
        if self.asleep and not self.consolidating and \
                self.drives.level("sleep_pressure") <= self.knobs.get("wake_below", self.loop.get("wake_below", 0.15)):
            self.asleep = False
            self._woke_at = self.t
            s.emit("brain", "woke")
            self._wake_intention(s, frame)

        if self.asleep and hasattr(world, "ambient"):
            # what still reaches a sleeping body (the dark, the cold, a blow); a strong stimulus (`rouse`
            # at or above loop.rouse_above) wakes it, whatever the pressure; the night's jobs finish awake
            felt = s.add(world.ambient())
            if any(float(e.data.get("rouse", 0.0)) >= float(self.loop.get("rouse_above", 0.5)) for e in felt):
                self.asleep = False
                self._woke_at = self.t
                s.emit("brain", "woke", roused=1.0)
                frame["roused"] = True
                self._wake_intention(s, frame)

        if self.asleep:
            # how restoring a sleep tick is: the `sleep_depth` knob (stress thins sleep)
            s.emit("brain", "sleep_tick", depth=round(self.knobs.get("sleep_depth", 1.0), 4))
            frame.update(asleep=True, action="sleep", consolidating=self.consolidating)
        else:
            # 1. sense
            sensed = s.add(world.sense())                    # percepts, and `contact` in shared worlds
            percepts = [e.data["text"] for e in sensed if e.kind == "percept"]
            # what the cerebellum learns on: the world's `focus` when it gives one (a big
            # world's full text almost never repeats), else the whole situation
            self._situation = next((e.data["focus"] for e in sensed if e.data.get("focus")),
                                   None) or " ".join(percepts)
            # and per action, when the world knows better (a move: what lies that way)
            self._focus_by_verb = next((e.data["focus_by_verb"] for e in sensed
                                        if e.data.get("focus_by_verb")), {})
            food_by_verb = next((e.data["food_by_verb"] for e in sensed if "food_by_verb" in e.data), {})
            food_far = next((e.data["food_far_by_verb"] for e in sensed if "food_far_by_verb" in e.data), {})
            food_cue = self._food_cue(food_by_verb, food_far)
            self._need_cues(sensed)
            if food_cue:
                frame["food_cue"] = food_cue
            addressed = any(e.data.get("attention") for e in sensed)
            if addressed:
                self._addressed_at = self.t
                if self.attention and self.loop.get("escalate_when_addressed", True):
                    self._offer("addressed", self._weight("addressed"))
            # listening: what others said or called stays in working memory, so it is
            # still there when System 2 is next free
            heard = False
            for p in percepts:
                if ' said: "' in p or "called out" in p or p.startswith("I heard a call from the"):
                    self.working.append(p)
                    heard = True
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
            oriented = (self._surprised_at is not None
                        and self.t - self._surprised_at <= int(self.loop.get("reply_within", 10)))
            novelty = max((self._novelty(p) for p in percepts), default=0.0)
            s.emit("brain", "novelty", amount=novelty)

            # 1b. recall from partial cues: what this moment reminds the mind of
            cue = " ".join(percepts) + " " + (self.goal or "")
            k1 = self._recall_k("recall_for_laya", 1)
            h_eps, h_notes = self._hunger_recall(k1)     # hungry: what comes to mind is eating (10ab)
            recalled = h_eps or self.memory.recall(cue, self.t, k1)
            nq = self._note_query(percepts, cue)
            if nq is not None:
                self.memory.cue_index.hear(nq)               # counted for the habituation at the next rebuild
            known = h_notes or self.memory.knowledge(cue, k1, query=nq)
            rec = ([e["id"] for e in recalled], [n["title"] for n in known])
            if rec != self._last_recall and (recalled or known):
                # what came back to mind, when it changes: the episodes' and the notes' own text
                self._log.append({"kind": "recall", "episodes": [{"id": e["id"], "t": e["t"], "text": e["text"]}
                                                                 for e in recalled],
                                  "notes": [{"title": n["title"], "recall": n.get("recall"), "text": n["text"]}
                                            for n in known]})
            self._last_recall = rec
            if recalled or known:
                s.emit("memory", "recalled", episodes=len(recalled), notes=len(known))

            # 2-3. feel, appraise
            verbs = {**world.verbs, **self.internal_verbs}
            # You cannot fall asleep without sleep pressure. With a `sleep_possible_above` knob
            # (step 10j) the gate moves with arousal: the sleep/wake flip-flop, where orexin
            # (motivated wakefulness) holds the mind awake until the pressure is higher.
            gate = self.knobs.get("sleep_possible_above", self.loop.get("sleep_possible_above", 0.0))
            if self.drives.level("sleep_pressure") < gate:
                verbs.pop("sleep", None)
            pull = self._goal_bias()
            if pull:
                s.emit("brain", "intention", pull=pull)      # an intention under way: orexin's cue
            # the orienting reflex stills the body (step 10e): while a thought started by a
            # surprise is in flight, System 1 only looks or waits, and does not fall asleep
            frozen = self._frozen()
            all_verbs = verbs                                # System 2 still sees every action
            if self.memory.cue_index is not None:
                self.verbs_seen.update(all_verbs)
            if frozen:
                verbs = {v: d for v, d in verbs.items() if v in frozen} or verbs
                frame["frozen"] = True
            state = self._render(percepts, novelty, recalled, known)
            if food_cue:
                state["_ctx"]["cue_bias"] = food_cue
            if self.gate is None:
                ap = self.fast.appraise(state, verbs, **self._mind_kw)
            else:
                s2_act = (thought.get("action") if thought and not thought.get("error")
                          and thought.get("action") in verbs else None)
                ap = self._fast_level(s, state, verbs, sensed, percepts, addressed, heard, s2_act, frame)
            action, conf = ap["action"], ap.get("confidence")
            salience = float(ap.get("salience", 0.0))
            if conf is not None:                             # a choice was weighed (carrying on weighs none)
                conf = float(conf)
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
            elif oriented:
                reason = "surprise"                          # my action did what I did not expect
            elif conf is not None and conf < self.knobs.get("escalate_below", 0.0):
                reason = "low confidence"
            elif (need >= self.loop.get("escalate_if_need_above", 1.1)
                    and self.t - self.last_escalated >= self.loop.get("need_escalation_every", 10)):
                reason = f"unmet need: {motive}"
            if self.attention:
                escalated = self._attend(s, percepts, all_verbs, conf, motive, need, frame)
                reason = None                                # the arbiter decided
            if (reason and not self.slow.busy
                    and self.drives.urgency("hunger") < self.loop.get("think_if_hunger_below", 1.0)):
                ws = self._workspace(percepts, verbs)
                self.slow.submit(ws, self.t)
                self._log_asked(ws, reason)
                self._addressed_at = None                    # System 2 now has the words
                if reason == "surprise":
                    self._surprised_at = None
                s.emit("brain", "escalated", reason=reason, confidence=conf)
                self.last_escalated = self.t
                escalated = reason

            # 5. integrate: a fresh thought may override the habit
            words = None
            by = "s1"                                        # who chose the act (habits learn from it)
            if thought and not thought.get("error") and thought.get("action") in verbs:
                action = thought["action"]
                by = "s2"
                words = thought.get("words") if action == "speak" else None
            if (action == "speak" and not words and self.loop.get("speech_via_slow")
                    and not (thought and thought.get("action") == "speak")):
                # speech is deliberate (Broca's area): System 1 only feels the urge to say
                # something; the words, or a plain call, are System 2's (step 10r)
                if self.attention:
                    self._offer("to speak", self._weight("to_speak"))
                frame["urge_to_speak"] = True
                action = "wait" if "wait" in verbs else "rest"
                by = "held"
            if self.drives.level("sleep_pressure") >= 1.0:
                action = "sleep"                             # collapse, whatever was chosen
                by = "forced"
            follows = by == "s1" and pull > 0 and action == self.goal_verb
            if self.habits and ap.get("raw_values") and action != "sleep":
                # the day's replay: what System 1 saw and valued, what was done and by whom;
                # the tick's reward is added once its events are in (step 7)
                self._habit_row = {"t": self.t, "state": {k: v for k, v in state.items() if not k.startswith("_")},
                                   "verbs": verbs, "raw": ap["raw_values"], "action": action, "by": by}

            # 6. act
            outcome_texts, outcome_novelty, events = [], 0.0, []
            if action == "sleep":
                self._fall_asleep(s)
            elif action == "rest":
                s.emit("brain", "rested")
            else:
                events = s.add(world.act(action, words=words) if words else world.act(action))
                had_effect = any(e.kind == "effect" for e in events)
                if any(e.kind == "ate" for e in events):
                    # what fed me: the meal's own words, for hunger's recall (step 10ab)
                    for e in events:
                        if e.kind == "outcome":
                            self.food_words += [w for w in tokens(e.data.get("text", ""))
                                                if w not in self.food_words]
                for e in events:
                    if e.kind == "outcome":
                        text = e.data.get("text", "")
                        self.working.append(text)
                        n = self._novelty(text)
                        outcome_novelty = max(outcome_novelty, n)
                        outcome_texts.append(text)
                        s.emit("brain", "novelty", amount=n)
                        ign = self.loop.get("novel_outcome_ignores")
                        if (ign is not None and n >= 1.0 and action not in ign
                                and (had_effect or not self.loop.get("novel_outcome_needs_effect", True))):
                            # never seen this come of an act (step 10x): novelty is itself
                            # rewarding (Bunzeck & Duzel 2006), less than a success; only when
                            # the act changed something (10z: 205 of s11's 248 were first failures)
                            s.emit("brain", "novel_outcome", amount=1.0)
                    elif e.kind == "effect":
                        self._effect(s, e.data.get("what", ""), frame)
            s.emit("brain", "acted", verb=action)
            if self.traces is not None and action not in ("sleep", "rest"):
                self._trace_hist.append({"t": self.t, "act": action, "near": nq if nq is not None else cue,
                                         "outcome": " ".join(outcome_texts),
                                         "effect": any(e.kind == "effect" for e in events)})
            if self.gate is not None and action != "sleep":
                self._note_act(action, events, outcome_novelty)
                if self.stuck is not None:
                    self._stuck_step(s, action, outcome_texts)
            sr = (self.loop.get("food_cue") or {}).get("search")
            if sr and action == sr.get("verb", "look"):
                self._searched_at = self.t
            self._spent(s, action, percepts, outcome_texts, frame)
            if self._habit_row is not None and follows and not frame.get("intention_spent"):
                self._habit_row["follows"] = True        # carried out the intention, and it still changed something
            whos = others_here(percepts)
            self._social_pending = (whos, action) if whos and action != "sleep" else None
            self.verb_counts[action] = self.verb_counts.get(action, 0) + 1

            # 6a. the cerebellum: compare what happened with what it predicted, learn
            prediction = None
            self._gate_surprise = 0.0
            if outcome_texts:
                prediction = self.predictor.observe(self._focus_by_verb.get(action, self._situation),
                                                    action, " ".join(outcome_texts))
                s.emit("cerebellum", "prediction", **{k: v for k, v in prediction.items() if k != "new"})
                if prediction["surprise"] >= float(self.loop.get("escalate_if_surprise_above", 1.1)):
                    self._surprised_at = self.t              # the outcome stays in working memory
                    self._gate_surprise = float(prediction["surprise"])   # the fast level hears it next tick
                    frame["surprised"] = prediction["surprise"]
                    if self.attention:
                        self._offer("surprise", self._weight("surprise") * prediction["surprise"])
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
                         action=action, confidence=None if conf is None else round(conf, 3), escalated=escalated,
                         salience=round(salience, 3), strength=round(strength, 3), wrote=wrote,
                         feeling=state["feeling"],
                         recalled=[e["id"] for e in recalled], known=[n["title"] for n in known],
                         prediction=prediction, outcomes=outcome_texts)
            if known and self.memory.cue_index is not None:
                frame["know_line"] = state.get("knowledge")          # the line Laya read (None: not shown)
                frame["know_sim"] = known[0].get("cue_sim")
                if known[0].get("act") or state.get("knowledge") is None:
                    frame["know_recall"] = known[0].get("recall")    # the recalled cue, shown or not
                    frame["know_act"] = known[0].get("act")
                    frame["know_pull"] = (state["_ctx"].get("act_bias") or {}).get(known[0].get("act"))
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
        before = {d.name: d.level for d in self.drives.active}
        self.drives.update(events, asleep=self.asleep)
        if self.traces is not None and not self.asleep and self._trace_hist and self._trace_hist[-1]["t"] == self.t:
            self._strong_moment(s, before)
        summary = summarize(events)
        if self.entrain is not None:
            self._entrain_step(summary, before)
        self.dials.update(self._signals(summary))
        self.drives.felt_signals = self.dials.levels()     # for the drives' `felt_gains` (3k)
        self.knobs = compute_knobs(self.knobs_cfg, self._signals(summary))
        if self._habit_row is not None:
            self._habit_row["r"] = habits.reward(self.habits, summary)
            # the tick's raw signals, so a reward can be re-tuned offline on real days
            self._habit_row["sig"] = {k: round(v, 4) for k, v in summary.items() if v}
            self._habit_store.add(self._habit_row)
            self._habit_row = None
        self._tag_recent()
        if self._habits_night is not None:
            frame["habits"] = self._habits_night
            self._habits_night = None
        if self._head_night is not None:
            frame["head_night"] = self._head_night
            self._head_night = None

        frame.update(
            thought=thought,                                 # the whole deliberation, for analysis
            consolidation=consolidation,
            review=review,
            **({"cues": cues} if cues else {}),
            thinking=self.slow.busy,
            goal=self.goal,
            task=self.task,
            motive=self.drives.motive(),
            open_contradictions=len(self.open_contradictions),
            drives=self.drives.snapshot(),
            dials=self.dials.snapshot(),
            knobs={k: round(v, 4) for k, v in self.knobs.items()},
            events=[e.kind for e in events],
        )
        if self._log:
            frame["log"] = self._log
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
            "goal_motive": self.goal_motive,
            "task": self.task,
            "task_set_at": self.task_set_at,
            "drives": self.drives.levels(),
            "dials": self.dials.levels(),
            "familiarity": self.familiarity,
            "verb_counts": self.verb_counts,
            "lifetime": self.lifetime,
            "distress_window": {n: list(w) for n, w in self.in_top.items()},
            "open_contradictions": self.open_contradictions,
            "effect_counts": self.effect_counts,
            "food_words": self.food_words,
            **({"head_trust": list(self._head_trust)} if self.head else {}),
            **({"entrain": self._entrain} if self.entrain else {}),
            **({"verbs_seen": self.verbs_seen} if self.memory.cue_index is not None else {}),
        })
        self.predictor.save()
