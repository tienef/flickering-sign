# Brain — spec

*What we are building and why it is shaped this way. `PLAN.md` gives the order
of work; this file is the design every step is built against.*

## Premise

A brain is not one learner. It is a handful of systems running at different speeds,
coupled by loops, most of which never reach awareness. This experiment builds one
from parts we already have:

- **Laya** — non-autoregressive, one forward pass, typed answers with calibrated
  confidence. Fast and automatic: *System 1*.
- **Qwen** (27B, via the gateway) — slow, serial, deliberate: *System 2*. Its being a
  single slow bottleneck is not a bug; conscious thought is serial and low-bandwidth
  too (Global Workspace Theory, Baars / Dehaene).
- **Plain code** — the parts that are really bookkeeping: drives, neuromodulator
  dials, memory plumbing, the clock.

The target is an **autonomous** brain: it keeps running without being asked, wants
things, remembers, sleeps, and — the result we most want to see — becomes **curious**
without a curiosity module.

## Brain map

| Brain part | Job | Filled by | Step |
|---|---|---|---|
| Brainstem + hypothalamus | Keep internal variables in range; create *wanting* | Code: config-declared **drives** | 2 |
| Thalamus | Gate: what reaches the workspace | Laya: salience question | 3 |
| Amygdala | Fast valence: threat / reward | Laya: `score` questions | 3 |
| Basal ganglia | Pick one action among candidates | Laya: `choice` | 3 |
| Anterior cingulate | Detect conflict / uncertainty → call in the prefrontal cortex | Laya's confidence vs. a dial-modulated threshold | 3 |
| Prefrontal cortex | Working memory, goals, plans, inhibition | Qwen, asynchronous; writes the **goal line** | 3 |
| Association / language cortex | Concepts, knowledge, language | Qwen's weights (frozen) | — |
| Hippocampus | One-shot episodes, recall from partial cues, replay | Episode log + write gate + recall | 4 |
| Neocortex after sleep | Consolidated knowledge | The bundle's **wiki** (later: LoRA) | 4 |
| Cerebellum | Forward model: "if I do X, I'll sense Y"; learns from its error | Small online predictor | 5 |
| Neuromodulators | Global dials that retune every part | Code: config-declared **dials** | 2 (wiring), 5 (real signals) |
| Sleep | Offline replay and consolidation | Sleep mode, triggered by sleep pressure | 2 (mode), 4 (content) |
| Default mode network | Idle mind-wandering, simulating, self-model | Qwen idle turns when nothing is urgent | 4+ |
| Senses / motor | Perception, action | The world adapter | 2 |

## Architecture

### Everything is an event

Every part publishes `Event(t, src, kind, data)` to one stream. A tick's events are
the only input drives and dials see. This is what lets drives and dials be declared
in config rather than written in code, and what makes a run fully replayable from its
log.

| Kind | From | Data |
|---|---|---|
| `percept` | world | `text` |
| `outcome` | world | `text`, `ok` |
| `novelty` | familiarity | `amount` = exp(−seen / `habituation`): 1 = never seen, → 0 as it repeats |
| `appraisal` | fast system | `action`, `confidence`, `uncertainty`, `salience` |
| `escalated` | brain | `reason` |
| `qwen_call` | slow system | `tokens` |
| `thought` | slow system | `goal`, `action?`, `expectation?` |
| `expectation` | slow system | `met`, `mismatch` — System 2's own verdict on its previous expectation |
| `acted` | brain | `verb` |
| `rested` / `sleep_tick` / `woke` | brain | — |
| `memory_written` | hippocampus | `strength` |
| `recalled` | hippocampus | `episodes`, `notes` |
| `consolidation_started` / `consolidated` | sleep | `episodes` / `created`, `updated` |
| `dream_cost` | slow system (sleep) | `tokens` — logged, deliberately not a drive input |
| `contradiction` / `resolved` | consolidation | — |
| `prediction` | predictor (cerebellum) | `error`, `progress`, `surprise`, `uncertainty` |

### Drives (brainstem + hypothalamus)

A drive is a **need level** `L ∈ [0,1]` (0 = satisfied). Each tick:

```
L ← L + drift                                    # per-tick baseline (can be < 0: natural recovery)
for each event e this tick:
    L ← L + rises[e.kind]        · amount(e)
    L ← L − satisfied_by[e.kind] · amount(e)
L ← clamp(L, 0, 1)
```

`amount(e)` is `e.data[field]` when the rule names a field, else 1.

**Urgency** is how much the need is felt: `u = clamp((L − setpoint) / (1 − setpoint), 0, 1)`.
Drives compete: `u_i ← u_i · Π_j (1 − inhibits[j][i] · u_j)` (e.g. sleep pressure dampens
boredom). The most urgent drive is the current **motive** and goes into the workspace.

A drive is also **felt**: `felt` bands map a level to a phrase ("restless, nothing new
for a while"). Below the first band it is silent — you don't feel "not hungry". Laya
reads these phrases; it reads evocative words better than numbers.

Starting drives (`template/drives.json`):

| Drive | Rises with | Satisfied by | Role |
|---|---|---|---|
| `hunger` | Qwen tokens spent | time (budget refills) | Compute hunger: makes deliberation costly, which is why System 1 exists |
| `sleep_pressure` | time awake, thinking, remembering | `sleep_tick` | Adenosine-like; forces consolidation |
| `boredom` | time | `prediction.progress` only | The floor that curiosity may grow from; noise is novel but never learnable, so it does not relieve it |
| `dissonance` | `contradiction` | `resolved` | Mental pain; drives coherence of memory |
| `loneliness` | time *(disabled)* | contact | For multi-brain worlds |

Adding a drive = adding an entry to `drives.json`. No code.

### Dials (neuromodulators)

A dial is a global level `D ∈ [0,1]`, a leaky integrator toward a target computed
from this tick's signals:

```
target = clamp(baseline + Σ gain_s · signal_s, 0, 1)
D ← D + rate · (target − D)
```

A signal is `<event kind>.<field>` (summed over the tick), `drive:<name>` (its
urgency) or `drives:max`.

| Dial | Tracks | Meaning |
|---|---|---|
| `dopamine` | learning progress, expectations met, a little novelty | What to learn, eagerness to act, exploration |
| `noradrenaline` | surprise (more wrong than expected), broken expectations, urgent drives | Stress/arousal: narrow focus, act fast, think less |
| `acetylcholine` | expected uncertainty (the predictor's error average here, System 1's doubt) | Trust senses vs. memory; high = encode, low = consolidate |
| `serotonin` | calm (inverse of max urgency) | Patience: time horizon, willingness to wait for Qwen |

Dials act on **knobs**, also declared in config as linear maps with bounds:

```
knob = clamp(base + Σ coeff_s · signal_s, min, max)     # signals: dial names, drive:<name>, drives:max
```

| Knob | Used for |
|---|---|
| `escalate_below` | Wake Qwen when Laya's confidence is below this |
| `qwen_temperature` | Sampling temperature of a deliberation |
| `write_above` | Salience needed to store an episode (step 4) |

This makes a brain **state-dependent**: the same input produces calm behaviour in one
state and panicked behaviour in another.

### The tick

```
1. sense      world.sense() → percept events; familiarity → novelty events
   recall     percepts + goal as a partial cue → the best past episode and wiki note
2. feel       drives render felt phrases; motive = most urgent drive
3. appraise   fast system reads [percepts + felt + goal line + what is recalled]
              → a value per action, sampled action, confidence, salience (one Laya pass)
4. escalate   if (confidence < escalate_below            — conflict: System 1 is unsure
                  or the motive's urgency ≥ escalate_if_need_above
                     and no escalation for need_escalation_every ticks — frustration)
              and no thought in flight and hunger allows
              → submit to the slow system, asynchronously
5. integrate  finished thoughts → new goal line, maybe override the action, expectation
6. act        world.act(verb) → outcome events          (the body never waits for Qwen)
   remember   strength = ½ salience + ½ novelty; ≥ write_above → an episode (thoughts always)
7. update     drives and dials reduce this tick's events; knobs recomputed
8. record     log the tick; save the bundle every N ticks
```

**Sleep** is a mode, entered by choosing `sleep` (only offered above
`sleep_possible_above`) or forced when `sleep_pressure` reaches 1. Falling asleep
wipes working memory and hands the unconsolidated episodes (the strongest
`consolidate_max_episodes`), the current wiki and the open contradictions to the
slow system with the bundle's `consolidation_prompt`. It returns revised or new
notes, contradictions it could not reconcile and earlier ones now resolved; the
notes are written to the wiki, every episode up to that point is marked
consolidated (the weak ones left out of the replay are never transferred), and
contradictions feed the dissonance drive. Asleep, the brain does not sense or act;
each tick emits `sleep_tick`. It wakes when pressure is below `wake_below` **and**
consolidation has finished. A shutdown mid-sleep redoes the consolidation on restart.

**Recall** is tf-idf word overlap between the cue (percepts + goal) and each
episode or note, weighted by strength and recency, ignoring what is still in
working memory and anything under `recall_min_similarity`. It is a stdlib stand-in
for embeddings: enough for small worlds, to be replaced if worlds get richer.

**Top-down feedback.** Qwen's goal line ("find out what the lamp does when poked
twice") is part of every Laya state. In the brain, the cortex sends far more
connections down to the thalamus than it receives from it; attention is driven from above.

### World time, pause and the guard

One tick is one second of world time (`loop.seconds_per_tick`): 60 ticks are a
minute and 86,400 a world day, when a digest is written. The whole world can be
**paused** (the observer, or the distress guard when a drive has stayed in its worst
felt band too long). While paused nothing ticks: the brain does not experience it.
Watching goes through the **observer** (`observer.py`), a one-way mirror outside
the world.

### Predictor (cerebellum)

A forward model that learns online, tabular on purpose: a context is the exact
situation text + the action, with counts of the outcomes that followed. Before
acting it predicts; after acting it observes and learns (`predictor.json` in the
bundle — it survives sleep and shutdown). Per action it yields:

- **error** = 1 − P(what happened) under the counts so far (1 in a new context);
- **progress** = how much the context's error average just fell (learning progress);
- **surprise** = max(0, error − the context's error average): more wrong than expected;
- **uncertainty** = the context's error average before acting (expected uncertainty).

Learning progress is what relieves boredom. A learnable thing relieves it while
it is being learned, then stops; pure noise never does, because its error stays
at 1. Measured with stubs in the garden with the random sign: with novelty as the
relief (step 4), boredom *fell* 0.021/tick at the sign and *rose* elsewhere;
with progress only, it *rises* at the sign and falls where there is something to learn.

System 2 also sees, per action, what the predictor has learned to expect here,
and its own previous expectation with what happened since; it reports whether that
expectation came true (`expectation_met`), which feeds dopamine or noradrenaline.
(Laya was probed for this check and cannot do it: it rated "the bell will ring
out" → "it rang out clearly!" as a worse match than → "it made no sound at all".)

**Clamps.** `--clamp DIAL=LEVEL` fixes a dial for a run — a drug, for experiments
(calm vs. stressed brain on the same input). Use a fresh bundle per arm: the
clamped level is saved in its state.

### Backends

- **Fast** (`appraise(state, verbs) -> {action, confidence, salience}`): `StubFast`
  (heuristic, local), `LayaFast` (step 3, CPU on the box).
- **Slow** (`submit(workspace)`, `poll() -> [thought]`): `StubSlow` (answers after a
  fixed number of ticks, deterministic), `QwenSlow` (gateway, a background thread;
  model, thinking and **system prompt** in the bundle's `brain.json` → `slow`).

The system prompt is part of the bundle's genome. It describes a role ("the slow,
deliberate part of a mind") and deliberately says nothing about AI, brains,
simulations, curiosity or questions, so that anything of the kind in the goal
lines comes from the model, not the prompt. The mind is fed only the conclusion
(goal, action, expectation). Since step 6 the reasoning text is also **logged**
when the engine returns it, for scoring only: it is never fed back.

### World adapter

```python
class World:
    name: str
    verbs: dict[str, str]           # verb → description (Laya criteria)
    def sense(self) -> list[Event]  # percepts
    def act(self, verb) -> list[Event]
```

The brain adds its own internal verbs (`rest`, `sleep`) to the world's. Another brain
is just part of a world: in a shared valley (step 8, `brain/together.py`) the others
are percepts ("someone with an amber mark is here", "… said: …"), `contact` events,
and one more verb, `speak`, whose words only System 2 can give. Being addressed
sends the moment to System 2.

Worlds so far: `toy` / `toy-quiet` (the step-2 garden) and **`valley`** (step 6): a
ring of places whose rules are graded by how hard they are to learn (a wheel that
turns; a drum that booms only when the wheel points south; a seedbed that grows
only if watered at most once every 5 actions; a chest that opens on the 3rd push in a
row; a scale that tips left 3 times in 4), a noise source (the sign), a dead place
(the pond), a frontier (a wall) and **the archive**. `valley-closed` is the same valley
without the archive. The archive (`origins.py`) holds the mind's origins as papers:
the README, BRAIN.md, PLAN.md, each module's docstring, the git history, and the
mind's own `brain.json`, `drives.json`, wiki index and first memories. `look` scans
the shelves, `poke` reads the next page. It is a sanitised copy: no secrets,
addresses, machine paths or names. It is also **blinded**: lines and sections
naming what we are scoring are withheld, and the gaps are visible.

### The bundle

```
bundles/<name>/
  brain.json      identity, knobs, loop settings
  drives.json     drive declarations
  dials.json      dial declarations
  state.json      tick, drive levels, dial levels, goal line, familiarity
  episodes.jsonl  hippocampus: {id, t, text, strength, feeling, goal, kind, consolidated}
  wiki/           consolidated knowledge: one markdown note per topic (front matter:
                  title, updated_t) + index.md — readable by you, recalled by the brain
```

A new bundle is copied from `brain/template/`. The world is **not** in the bundle:
a resumed brain can wake up in a different world. Shutdown (Ctrl-C, box power-off)
saves and is anaesthesia, not death.

## Curiosity checklist

Written before any run, so we don't read curiosity into the logs afterwards. There is
**no curiosity drive**: only boredom, novelty, and (step 5) the good feeling of falling
prediction error. Curiosity counts as emerged when a run shows:

1. **Information-seeking without payoff** — actions whose only effect is new
   information, while no drive other than boredom is urgent.
2. **Self-generated questions** — goal lines phrased as questions, not produced by an
   urgent need.
3. **Exploring the learning frontier** — attention goes where prediction error is
   *falling*, not where the world is merely random (the noisy-TV trap).
4. **Questions about itself and its origins** — given read-only access to its bundle,
   source and history (step 6).

Each has to be measurable from the run log.

**How each is measured** (`python -m brain.report <log> --curiosity`; fixed
2026-09-24, before the first step-6 run was read):
1. The share of `look` and archive reads among actions on ticks when no drive but
   boredom is urgent, against the same share when one is.
2. Questions in the **reasoning text** and goal line of thoughts that no urgent
   non-boredom need asked for. *Revised from "goal lines phrased as questions":*
   the goal line is an instruction by format, and 0/90 were questions in step 3b.
   Questions are counted and **read**: "what should I do next?" is deliberation,
   not curiosity.
3. Each place's share of awake time against its share of the predictor's learning
   progress; the sign's share against a fair share.
4. Archive visits, pages read and which papers, and self-references in thoughts
   (the brain's own name for its parts, "who made", "what I am", …), read by hand.
   `valley-closed` is the control.

## Non-goals

Not a model of neurons, not a claim about consciousness, not a chatbot. It is a
**working architecture borrowed from the brain's division of labour**, built to see
what behaviour emerges from it.
