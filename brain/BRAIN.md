# Brain — spec

*What we are building and why it is shaped this way. `PLAN.md` gives the order of work; this file is the design
the code is built against. Rewritten on 2026-09-30 for the three levels of control, which replaced an earlier
architecture (Laya as System 1, an LLM as System 2; v0.1 of this repository).*

## Premise

A brain is not one learner. It is a handful of systems running at different speeds, coupled by loops, most of
which never reach awareness. This experiment builds one from parts we have:

- **An LLM** (Qwen3.8-27B uncensored, FP8, on vLLM) read two ways: in **one pass**, the probability of each
  option's letter at the first token (the middle level: what to do now), and **thinking** (the slow level:
  intention, speech, and at sleep the wiki).
- **Laya** (non-autoregressive, one forward pass): a small head per mind that learns, at sleep, the choices the
  middle level made that day, and may later make the familiar ones alone.
- **Plain code** for what is really bookkeeping: drives, hormone dials, the gate that says when a moment calls for
  a choice, memory plumbing, the cerebellum's predictions, the clock.

The target is an **autonomous** mind: it keeps running without being asked, wants things, remembers, sleeps, and
lives with others on a small island. Nothing is programmed of curiosity, love, friendship, empathy, morals or
cooperation: they may emerge from interactions and their consequences, or not. The result most wanted is
curiosity without a curiosity module.

## Three levels of control

Human parallels: LeDoux's low and high roads, Norman & Shallice (routines, contention scheduling, the supervisory
system), Rasmussen's skill / rule / knowledge, Stanovich's autonomous / algorithmic / reflective minds. Each level
is slower and richer than the one below and is asked only when the one below cannot settle the moment.

| level | who | asked | what it does | adapts by |
|---|---|---|---|---|
| **fast** | generic code (the gate, the carry-on rule) + a Laya head per mind | every tick | says whether the moment calls for a new choice; if not, the body carries on; the head proposes on the flagged ticks | the head is distilled at sleep from the middle level's choices (automatisation, ACT-R's proceduralisation) |
| **middle** | the LLM, one pass, no thinking | when the gate flags the tick | reads the situation, feelings, intention, what comes to mind; the act is drawn from its letters' odds | through what it reads (notes, cues, intention), never its weights |
| **slow** | the same LLM, thinking | when the arbiter gives it the floor | deliberates: intention and task, expectation, speech; at sleep: review, consolidation into the wiki, cues | through its notes and cues |

Why this shape (measured on the earlier architecture): Laya cannot be the one who chooses (flat values, follows lures
naming a verb, unmoved by feelings); the LLM read in one pass does not follow lures, follows the slow level's
intention (act on top 0.78-0.80), is moved by feelings, generalises rules to unseen kinds, 0.13-0.17 s a call. It
cannot be called on every tick of several minds (a hybrid model with no prefix caching: ~12 calls/s at most), so a
gate in code decides when a moment is worth a choice: "what is in front / held changed" alone fires on ~21% of
ticks and flags 87-100% of the decisive moments.

## Brain map

| brain part | job | filled by |
|---|---|---|
| Brainstem + hypothalamus | keep internal variables in range; create *wanting* | code: config-declared **drives** |
| Neuromodulator nuclei, glands | global states that retune every part | code: config-declared **dials** (hormones) and the **knobs** they move |
| Thalamus + salience network (bottom-up) | what catches attention: a change, a failure, a surprise, an alarm | the fast level's **gate** (code) |
| Basal ganglia routines | keep doing what works; follow the plan; wait | the **carry-on** rule (code) |
| Striatum, habits | familiar choices made without the cortex | the per-mind **fast head** (Laya), distilled at sleep |
| Cortex, fast read | recognise the situation and what it affords, in one pass | the **middle level** (the LLM, letters) |
| Anterior cingulate | conflict: the options are close | the middle level's split vote (the drawn act's probability below `escalate_below`) bids for the slow level |
| Salience network (arbiter) | who gets the slow level, what may interrupt it | code: the **attention** block |
| Prefrontal cortex | intention, task, expectation, inhibition | the **slow level** (the LLM thinking); its intention is read by the middle level |
| Broca's area | the words | the slow level only (the body feels an urge to speak) |
| Hippocampus | one-shot episodes, recall from partial cues | episode log, write gate, recall (tf-idf) and cues (bge-small, rebuilt at sleep) |
| Neocortex after sleep | consolidated knowledge | the bundle's **wiki**: notes as hypotheses, reviewed each night |
| Cerebellum | forward model, surprise, learning progress | a tabular **predictor** per action and context |
| Sleep | offline replay, consolidation, automatisation | review, consolidation, cues (the slow level, no thinking), the head's distillation (Laya) |
| Default mode network | thought nobody asked for | daydreams (an `idle` bid; off in the template) |
| Senses, motor | perception, action | the world adapter |

## Everything is an event

Every part publishes `Event(t, src, kind, data)` to one stream. A tick's events are the only input drives and dials
see: this is what lets them be declared in config rather than written in code, and what makes a run replayable from
its log.

| kind | from | data |
|---|---|---|
| `percept` | world | `text`, and on some percepts `focus`, `focus_by_verb`, `attention` (addressed), food hints |
| `outcome` | world | `text`, `ok`, `changed` (the land, the body's holdings, place or facing, or the mist) |
| `effect`, `ate`, `contact`, `chill`, `warmth` | world | `what`, `amount` |
| `novelty` | brain | `amount` = exp(−seen / `habituation`): 1 = never seen |
| `novel_outcome` | brain | an act gave an outcome never met before, and changed something |
| `gate` | fast | `salience` of a flagged tick |
| `appraisal` | fast | `action`, `confidence`, `uncertainty` (a choice was weighed; carrying on weighs none) |
| `middle_call`, `middle_error` | middle | `tokens`, `seconds` |
| `escalated`, `interrupted` | brain | `reason`, `level` |
| `qwen_call` | slow | `tokens`, `seconds` (a thought; feeds compute hunger) |
| `thought`, `task_set`, `task_done`, `expectation` | slow | `goal`, `expectation`; `met`, `mismatch` |
| `daydream`, `slow_error` | slow | — |
| `intention`, `intention_spent`, `goal_faded`, `task_faded` | brain | `pull`, `verb` |
| `acted`, `rested`, `sleep_tick`, `woke` | brain | `verb`; `depth` |
| `effectance` | brain | `amount`: my act changed something, scaled by age and habituation |
| `memory_written`, `recalled` | memory | `strength`; `episodes`, `notes` |
| `review_started`, `consolidation_started`, `consolidated`, `cues_started`, `note_retired`, `contradiction`, `resolved` | sleep | counts |
| `dream_cost` | slow (sleep) | `tokens`, `seconds`: logged, deliberately not a drive input |
| `head_night` | brain | `accepted` |
| `prediction` | cerebellum | `error`, `progress`, `surprise`, `uncertainty` |
| `opioid`, `adrenaline`, `cortisol`, `oxytocin`, `ghrelin` | dial | `level`: hormones the body feels (`loop.dials_as_events`, last tick's level) |

## Drives (brainstem + hypothalamus)

A drive is a **need level** `L ∈ [0,1]` (0 = satisfied). Each tick:

```
L ← L − decay · L                                # a leak toward 0, if declared (a state that fades: frustration)
L ← L + pace · drift                             # per-tick baseline (drift_asleep while asleep; can be < 0)
for each event e this tick:
    L ← L + pace · rises[e.kind] · amount(e)      # pace 1 unless the brain entrains the drive
    L ← L − satisfied_by[e.kind] · amount(e)
L ← clamp(L, 0, 1)
```

`amount(e)` is `e.data[field]` when the rule names a field, else 1. **Urgency** is how much the need is felt:
`u = clamp((L − setpoint) / (1 − setpoint), 0, 1)`. Drives compete: `u_i ← u_i · Π_j (1 − inhibits[j][i] · u_j)`
(sleep pressure dampens boredom). The most urgent drive is the **motive**. A drive is **felt** through bands that map
a level to a phrase ("restless; nothing new has happened in a while"); below the first band it is silent (a drive's
`felt_gains` add dial levels to the level for its phrase only: sleepiness is felt from the pressure and melatonin). The
middle and slow levels read these phrases, never the numbers.

The template's drives (`template/drives.json`):

| drive | rises with | eased by | role |
|---|---|---|---|
| `hunger` | the slow level's tokens | time, rest, adrenaline | compute hunger: thinking is costly (the middle level's calls are free) |
| `sleep_pressure` | time awake, thoughts, episodes, novelty | `sleep_tick` (× the `sleep_depth` knob) | adenosine-like; forces sleep, hence consolidation |
| `boredom` | time | learning progress, effectance, opioids | the floor curiosity may grow from; raw novelty does not relieve it (noise is novel, never learnable) |
| `dissonance` | `contradiction` | `resolved` | mental pain; drives coherence of memory |
| `loneliness` | time | `contact` (by kind), opioids, oxytocin | on: minds live together |
| `food` | time (half as fast asleep), ghrelin, cortisol | `ate` (× the food's amount) | the body's hunger: two meals a waking day |
| `cold` | `chill` (outside at night) | `warmth` (a roof, a fire) | the body's warmth |
| `thirst` | time (0.008 a tick, half asleep: faster than food) | `drank` × 2 its amount (salt water has a negative amount: worse) | the body's water; off in the template, on in the island's overlay |
| `pain` | `hurt` (a blow's strength) | heals by itself (negative drift, faster asleep), opioids | the body's pain (strike); its weakness lives in the world's body; off in the template, on on the island |
| `frustration` | `thwarted` (below) | `gained`, a little `went`; `decay` 0.02 (half-life ~35 ticks) | a blocked goal (Amsel; Dollard, Berkowitz); its effects are gains elsewhere: noradrenaline, adrenaline, cortisol, `choice_temperature`, `interrupt_margin`, the carry-on letting go, the `stuck` bid |

Adding a drive is adding an entry to `drives.json`; no code. A drive may be `enabled: false` in the template and
turned on by an overlay (`variants/island.json` turns on thirst and pain).

**Frustration's events** (genome block `stuck`, `brain.py` `_stuck_step`). An act that changed nothing (the world's
`changed`, as the gate's `fail`) while something is wanted (a goal, a task, or a need at `active_above` 0.5) emits
`thwarted`, of amount `same` (1.5) when the same act meets the same thing within `same_within` (10) ticks, else 1; an
act with an effect emits `gained`, one that changed something else `went`. While frustration is felt (from its first
band) and the same act has failed at least twice, the slow level reads what keeps failing, counted: *"I have tried to
<act> <n> times; <the outcome>"* (`_stuck_line`). What the mind does about it is not coded.

## Dials (hormones) and knobs

A dial is a global level `D ∈ [0,1]`, a leaky integrator toward a target computed from this tick's signals, with
its own rate up and (optionally) down:

```
target = clamp(baseline + Σ gain_s · signal_s, 0, 1)
D ← D + rate · (target − D)                      # rate_down when target < D
```

A signal is `<event kind>.<field>` (summed over the tick), `<event kind>` (a count), `drive:<name>` (urgency),
`level:<name>` (a drive's level), `drives:max`, or another dial. Each hormone is modelled on its known human effects
(the gains and their sources are in `template/dials.json`):

| dial | rises with | acts on |
|---|---|---|
| `dopamine` | learning progress, expectations met, effectance, a little novelty | explore (choice temperature up), write episodes more readily |
| `noradrenaline` | surprise, broken expectations, urgent drives, adrenaline, frustration | arousal: gate lower, choice narrower, interrupt sooner, rest shorter |
| `acetylcholine` | expected uncertainty, a split vote | attention to the senses: gate lower, write more |
| `serotonin` | calm (cortisol lowers it) | patience: harder to interrupt, longer rest after a thought |
| `orexin` | surprise, an intention under way, bodily need, ghrelin; melatonin and sleep lower it (on the clock, silent in sleep) | motivated wakefulness: holds the sleep gate shut, keeps a body from dozing |
| `opioid` | eating, a task done, an expectation met, understanding | "liking": soothes boredom and loneliness, calms |
| `adrenaline` | surprise, broken expectations, a task done, frustration | tags the moments just before it for memory |
| `cortisol` | a need that lasts, acts that change nothing, frustration, broken expectations; sleep and its own level lower it (negative feedback) | stress: gate higher (control shifts to habit), fewer memories recalled, thinner sleep |
| `oxytocin` | contact | bonding: eases loneliness, calms, company remembered more strongly |
| `ghrelin` | the energy deficit, before hunger is felt; falls after a meal | makes food in sight wanted; raises food hunger |
| `melatonin` | the world's `dark` (full at night, half by a fire, a little in the evening); falls fast in light | the circadian signal: does not tire (that is sleep pressure) but opens the sleep gate in the evening and holds sleep through the night (`sleep_possible_above`, `wake_below`, `doze`), damps the restless search; worlds without `dark` leave it at 0 (Borbély's two processes) |

Dials act on **knobs**, declared in config as linear maps with bounds:
`knob = clamp(base + Σ coeff_s · signal_s, min, max)` (signals: dial names, `drive:<name>`, `drives:max`,
`<event>.<field>`). This makes a mind **state-dependent**: the same input gives calm behaviour in one state and
narrowed, hurried behaviour in another.

| knob | used for |
|---|---|
| `gate_above` | the salience at which the fast level flags a tick (noradrenaline and acetylcholine lower it, cortisol raises it) |
| `choice_temperature` | the middle level's draw over its letters' odds (1 = the model's own; dopamine and frustration up, noradrenaline and cortisol down) |
| `escalate_below` | a drawn act with a probability below this is a conflict: a bid for the slow level |
| `qwen_temperature` | the slow level's sampling temperature |
| `write_above` | the strength needed to store an episode |
| `interrupt_margin` | how much a bid must beat the thought in flight to interrupt it (frustration widens it) |
| `rest_after_thought` | ticks the chronic bids are not heard after a thought lands |
| `sleep_possible_above` | the sleep pressure above which `sleep` is offered |
| `doze` | the chance an idle tick is `sleep` when sleep is offered (melatonin and the pressure raise it, orexin lowers it) |
| `wake_below` | asleep, the mind wakes when sleep pressure falls to it (0.15; melatonin takes it below 0 in the dark, so a night's sleep lasts until dawn) |
| `restless` | the chance an idle body searches instead of waiting (hunger, thirst, ghrelin and orexin raise it; melatonin and sleep pressure damp it: food-seeking agitation) |
| `think_effort` | the effort a thought gets, read with `slow.effort` (noradrenaline helps up to 0.6 then hinders; compute hunger and a pressing need make it quick) |
| `emotional_tag` | strength added to the last episodes by an adrenaline surge |
| `recall_scale` | how many memories come back |
| `sleep_depth` | how restoring a sleep tick is |
| `social_tag` | strength added to an episode with someone in it |
| `food_cue_gain` | a pull (added to the log-probabilities) toward the acts that lead to food in sight, times hunger |

`--clamp DIAL=LEVEL` fixes a dial for a run: a drug, for experiments (a fresh bundle per arm).

## The fast level (`brain.py`: `_gate_signals`, `_carry_on`, `_fast_level`; genome block `fast`)

**The gate.** Each tick, generic signals fire, each at its genome weight (`fast.gate.signals`); the tick's
salience is the strongest; the tick is **flagged** when the salience reaches the `gate_above` knob. Nothing in them
is of any world:

| signal | fires when | weight in the template |
|---|---|---|
| `woke` | the first awake tick (a start, a waking) | 1.0 |
| `front` | what is in front or held changed (the world's `focus`, else the whole situation) | 1.0 |
| `addressed` | spoken to | 1.0 |
| `surprise` | the cerebellum's surprise at the last act (× its size) | 1.0 |
| `alarm` | a drive crossed into a higher felt band (a crossing, not a level) | 0.9 |
| `effect` | the last act changed the world or what the body holds | 0.8 |
| `need` | a drive at urgency ≥ `need_above` (0.5), nothing flagged for `need_every` ticks (5, down to `need_every_min` 3 as the urgency reaches 1) (× urgency) | 0.8 |
| `heard` | words or a call heard | 0.7 |
| `intention` | the slow level gave a new intention or task | 0.7 |
| `others` | who is in sight changed | 0.6 |
| `fail` | the last act changed nothing, the first time in a row (not `wait`, `look`, `speak`) | 0.6 |
| `long` | nothing flagged for `long_after` (30) ticks: the routine is checked now and then | 0.5 |
| `novel` | the last act's outcome was never met before | 0.4 (flags only when arousal lowers the gate) |
| `relief` | what relieves a felt need came into view (`loop.need_cues`, below) (× the drive's urgency) | 0.8 |

`need` ignores the drives in `need_ignores` (`frustration`, `sleep_pressure`: one is said by its own bid, the
other by sleep's offer). **Need cues** (`loop.need_cues`, `_need_cues`, 3n I5): for each, the world's `<by>` on
the percept (`drink_by_verb` for thirst, `food_by_verb` for food: which act would relieve the need on what is in
view), the drive it serves and the level from which it counts; a cue newly in view while the drive is at or above it
fires `relief` and may bid for the slow level (thirst: 0.6 × urgency). An innate orienting to what the body lacks
(Berridge), as hunger has for food.

**Who chooses the act** (logged per tick in `frame["fast"]`: flag, why, by, how it carried, seconds, letter mass):
1. a thought of the slow level that lands this tick with an action: that action (`by: slow`);
2. a flagged tick: the head proposes; in `decide` mode it acts alone when sure and trusted (`by: head`); otherwise
   the middle level chooses (`by: middle`) and its whole distribution becomes a label for tonight;
3. otherwise, or when the middle level fails: **carry on** (`by: carry`): repeat the last act while it changes
   something (walking on), else the intention's act (let go when that act just changed nothing and frustration is
   at `fast.carry.stuck_above` 0.4 or more), else, with the chance the knob `restless` gives (hunger,
   thirst, ghrelin and orexin raise it, melatonin and sleep pressure damp it: food-seeking agitation), a search:
   (before the search, when `sleep` is offered: with the chance the knob `doze` gives, `sleep`: falling asleep with
   nothing to do in the dark is the sleep switch flipping, not a decision)
   one of `fast.carry.search`'s verbs (the moves), the same way until it changes nothing; else the idle act
   (`wait`, then `rest`).

**The head** (`fast.head`, defaults `habits.HEAD_DEFAULTS`): the mind's own copy of a Laya decision head (the
`english` checkpoint, choice mode, fp16 on the card, no salience question). Mode `observe` (the template): it
proposes on the flagged ticks, is scored against the middle level's top act, and never acts. Mode `decide`: it acts
alone when its choice's probability is ≥ 0.9 and it agreed on ≥ 0.8 of the last 50, and 10% of those ticks are
still asked, so its record goes on. At each sleep, `habits.distill_night`: 20% of the day's labels held out, the
rest trained (cross-entropy to the soft labels) with older nights' labels interleaved; the night is kept only if
agreement on every held-out label kept so far drops by no more than 0.02, else rolled back with its reason.

## The middle level (`backends.QwenFast`; genome block `middle`)

One chat call, no thinking, `max_tokens` 1, `top_logprobs` 20. The system prompt: *"You are the fast, intuitive
part of a mind. You do not deliberate: you read what the mind senses, how it feels and what it is aiming for, and
answer at once with what it does next. Reply with the letter of one option and nothing else."* The user message
is the mind's state, one line per field, then the question and the actions as lettered options:

```
Situation: <percepts> A moment ago: <the last `recent_for_fast` working-memory items>
Feeling: <felt phrases, or "nothing pressing">
Goal: <the intention, or "no particular goal">
Task: <the task>
Memory: I remember: <the best recalled episode>
Knowledge: I know: <the recalled cue line, or the note's first sentence>

Given what this mind senses, how it feels and what it is aiming for, what does it do next?
Options:
A. north: go north (or turn to face it, if something is in the way)
...
Answer with one letter.
```

The letters' log-probabilities at the first token are the model's odds over the acts (an option outside the top
20 is put below all of them); code biases may be added to the log-probabilities (the food cue under ghrelin; the
intention's and a cue's pulls are 0 in the template: the model follows the intention as text); the act is drawn at
the `choice_temperature` knob. Confidence = the drawn act's probability under the model's own odds: a split vote is
the conflict the arbiter hears. Letter mass (how much of the reply went to an option at all) is logged. A failed call
returns no act and the body carries on. Prompt and question are the benched ones (`probe/one_model.py`): changing
them changes the experiment.

## The slow level (`backends.QwenSlow`; genome block `slow`)

The same model on the same port, thinking, one job at a time on a background thread (two workers: an interrupting
thought starts while the one it supersedes finishes). The body never waits for it.

**The effort of a thought** (`slow.effort`, the `think_effort` knob, 3n I6): the knob + `stakes_gain` (0.5) × the
value of the bid that asked it, against bands: below 0.25 `none` (no thinking), below 0.7 `low`, above `medium`
(its own cap, `max_tokens` 3000: the thinking and the reply share it). `dark_only: ["medium"]`: medium only while
the body senses the dark, else `low` (on the island medium took 40-90 s, 80-180 ticks of carrying on; the night's
awake thoughts are where a long one costs the body least). A thought whose thinking used up its cap before any
reply is asked once more without thinking (a floor of 600 tokens). Benched alone on the engine: none 2.4 s, low
19.2 s, medium 26.7 s (p50). The mind's effort is in the effort, not in the prompt (Shenhav et al.'s expected value
of control; noradrenaline's inverted U). The night's jobs keep `think_at_night`.

**A thought** is asked by the arbiter. It is given the workspace: what is sensed now, working memory (oldest first),
up to 3 recalled episodes and 3 notes, open contradictions, what the cerebellum expects of each action here, its own
last expectation and what happened since, feelings, the urge to speak if any, the task, the intention, and the
actions. The system prompt describes a role ("the slow, deliberate part of a mind ... a faster part ... has called
on you") and deliberately says nothing about AI, brains, simulations, curiosity or questions. The reply's shape is
imposed by vLLM after the thinking (`response_format` json_schema, `slow.json_schema`): `task`, `task_done`,
`goal` (the new **intention**), `action` (one of the body's actions, or null), `words` (when speaking),
`expectation`, `expectation_met`. The reasoning text is logged for scoring only, never fed back.

**Integration** when it lands: the goal becomes the intention (fades after `goal_ttl`, 30 ticks), the task lasts the
waking period (until done, replaced, or `task_ttl`), the expectation is kept and checked by the next thought
(`expectation_met` feeds dopamine or noradrenaline), the action is done this tick, the decision is written as an
episode ("I thought it over and decided: ..."). A thought superseded by an interruption is dropped when it lands
(logged in full as `thought_cut`): it answers a moment that is gone, unless it **lands late** (`loop.intentions`,
below; logged as `thought_late`).

**An intention outlives its moment while its motive holds** (`loop.intentions`, 3n Q1). Immediate: a cut thought
still lands if it was asked within `late_within` (240) ticks, the mind's motive (its most urgent drive) is the one it
was asked under, and no thought has landed since (Zeigarnik; intention superiority, Goschke & Kuhl). Across sleep
(`through_sleep`): the goal does not fade while asleep; at waking it is kept, held afresh, if its motive's urgency is
still at `keep_above` (0.5) or more, else let go; falling asleep with a goal writes the episode *"Before I slept, I
still meant to: ..."* with strength 1 + `unfinished_tag` (0.3), so the night's consolidation sees it among the first
(memories relevant to the future are consolidated first, Wilhelm et al.; Scullin & McDaniel).

**Speech** is the slow level's only: when the middle level or the carry-on picks `speak` without words, the body
waits and feels an urge to speak, which bids for the slow level; the words come with its next thought.

**At sleep** (without thinking in the template, `think_at_night: false`: a consolidation takes ~7-11 s instead of
100+ s, while the body's needs rise asleep): review, consolidation, cues (below).

## The arbiter (salience network; genome block `attention`)

Who gets the slow level, and what may interrupt it. Every source **bids** on one scale, weighted by the genome:

| source | bid (template) |
|---|---|
| a drive entering its worst band | `worst_band` 1.0 |
| the cerebellum's surprise | `surprise` 0.9 × its size |
| being spoken to | `addressed` 0.8 |
| the urge to speak | `to_speak` 0.7 |
| an urgent need (`escalate_if_need_above`), every `need_escalation_every` ticks | `need` 0.6 × urgency |
| the intention's act stopped changing anything (spent) | `intention_spent` 0.6 |
| food in sight while hungry | `loop.food_cue.bid` 0.6 × hunger |
| what keeps failing, while frustration is the motive | `stuck` 0.8 × urgency (the bid names the stuck line) |
| water newly in view while thirsty | `loop.need_cues` bid 0.6 × thirst |
| a split vote of the middle level | `low_confidence` 0.3 × (1 − confidence) |
| nothing asks and the slow level has been free a while | `idle` (a daydream; 0 = off in the template) |

The strongest bid waits; the thought in flight holds the priority of the bid that started it; both fade with age
(`half_life` ticks). A free slow level takes the waiting bid; a busy one is interrupted when the bid beats the
thought's level by the `interrupt_margin` knob. Some sources only wait (`cannot_interrupt`: the split vote), and a
source never interrupts its own thought. Right after a thought lands, the chronic sources (`rest_blocks`: split
vote, unmet need) are not heard for `rest_after_thought` ticks: the body carries out the intention before the mind
thinks again. Too drained (compute hunger above `think_if_hunger_below`), bids wait. While a thought started by a
surprise is in flight, the body may only `look` or `wait` for at most `freeze_max` ticks (the orienting reflex), and
that thought gets a smaller budget.

**An intention ends when its act stops changing anything:** done again, the same situation gave the same outcome.
The goal line stays, its act is no longer followed, and the slow level is asked for the next step.

## Memory

**Working memory** (8 items, not persisted): recent outcomes and what was heard. Sleep wipes it, with the task.

**Episodes** (hippocampus). A tick's strength is `½ salience + ½ novelty` (the gate's salience, the newest of what
was sensed or came of the act); at or above the `write_above` knob the moment is written:
"<percepts> I chose to <act>. <outcomes>". A thought's decision is always written. Oxytocin adds strength to a moment
with someone in it (`social_tag`); an adrenaline surge adds strength to the last `tag_window` ticks' episodes
(`emotional_tag`). Strength decides what sleep replays and how readily an episode comes back.

**Recall** from partial cues: percepts + intention against every episode and note (tf-idf word overlap, weighted by
strength and recency, ignoring what is still in working memory, under `recall_min_similarity`); notes come back
through their **cues** (`loop.recall_index`, `recall_index.py`): at sleep each note gets the moments it will matter
("the thing sensed then") and a line to recall; embedded with `bge-small-en-v1.5`, matched against what is near
(the percepts starting "In front of me:", "I carry", ...), above `min_similarity` 0.7; a cue that fires on more than
15% of the moments habituates and leaves the index. Hunger brings back the memories of eating (state-dependent
recall, from the words of past meals). The middle level reads the best episode and the best note's line; the slow
level gets up to 3 of each (× `recall_scale`).

**Traces written in the moment** (genome block `traces`, off in the template; the island arm's overlay
`variants/traces.json` turns it on; `brain.py` `_strong_moment`). When an act relieves a body need at once (its
level falls by `relief_above` beyond the drift) or makes one worse (rises by `harm_above`), a provisional cue goes
into the index at once: `when` = what was near, `recall` = "Here: <the world's words> <the drive's relieved /
worsened phrase>", the act for a relief, none for a harm; a relief also credits the last act that changed what is
held or the world within `chain_ticks` (taking, then eating) when the relief used something up: then only that
chain's trace is written, at the source act's place, act the source's. It comes back like a note (the middle level's "I
know" line) until the index is rebuilt after the night's consolidation, which reads the day's traces and keeps
what it rewrites into notes. One-trial learning (Garcia; the hippocampus's fast encoding, complementary learning
systems), sorted by sleep. Benched: thirsty at the sea, P(drink) 0.95 -> 0.08 with the sea-water trace; the same
line where it does not apply moves nothing.

**The wiki** (neocortex): one markdown note per topic, readable by us, recalled by the mind. Notes are hypotheses.

## Sleep

Entered by choosing `sleep` (offered above the `sleep_possible_above` knob), by dozing off on an idle tick (the
`doze` knob), or forced when sleep pressure reaches 1. **Entrainment** (genome block `entrain`, `brain.py`
`_entrain_step`): the body measures its world's day (the run of light between two darks) and its sleep pressure's
rise a waking tick, and at each nightfall moves the drive's `pace` (which scales its drift and rises) toward the
one that fills it to `target` by nightfall; it starts from `init` (the island's newborns inherit 0.6) and refines
it; a world without `dark` leaves it there.
Falling asleep wipes working memory and the task, lets effect habituation recover by half, and starts the night's
jobs: the **head's distillation** (synchronous, ~3-15 s on the card, as the mind falls asleep) and, on the slow
level, in order:
1. **review** (when the wiki has notes): the slow level says whether the day's experiences bore each note out, went
   against it, or said nothing about it; the verdicts are kept in the note (the last `review_window`); a note failed
   at least `retire_after` times, and more often than it held, is retired to `wiki/retired/` (kept, never recalled).
   The world does the sorting: the prompt says nothing about what makes a good note;
2. **consolidation**: the strongest unconsolidated episodes (up to 60), the surviving notes and the open
   contradictions; it returns notes added or revised, contradictions it could not reconcile, and those now
   resolved; contradictions feed dissonance;
3. **cues** for the notes this night touched, each with the body's act by which the mind did what the line says,
   named from its own episodes ("What my acts did"), or `none`; the index is rebuilt when they land (the mind may
   already be awake).

Asleep, the mind does not sense or act; each tick emits `sleep_tick`. It wakes when sleep pressure is below
`wake_below` **and** the consolidation is done. A shutdown mid-sleep redoes the consolidation on restart; shutdown is
anaesthesia, not death.

## The cerebellum (`predictor.py`)

A forward model that learns online, tabular on purpose: a context is the situation (the world's `focus` for that
action when it gives one: a move learns on what lies that way) + the action, with counts of the outcomes that
followed; it survives sleep and shutdown (`predictor.json`). Per act: **error** = 1 − P(what happened);
**progress** = how much the context's error average just fell (learning progress, what relieves boredom: a
learnable thing relieves it while it is being learned, noise never); **surprise** = max(0, error − the context's
average); **uncertainty** = the context's average before acting. With `predictor_prior`, a new context is judged
against what the action does in general (by the outcome's shape), so a door that opens after sixty pushes that did
nothing is a surprise. Surprise above `escalate_if_surprise_above` bids for the slow level and fires the gate's
`surprise` signal on the next tick. In shared worlds it also learns how each other mind responds to what this one
did ("with <mark>"), shown to the slow level.

## The tick (`Brain.tick`)

```
0. hormones the body feels become events; the intention and the task fade if held too long
1. integrate the slow level's finished jobs: a thought (intention, task, expectation, act), a daydream, the night's jobs
2. asleep: wake if pressure < wake_below and the consolidation is done; else a sleep tick, stop here
3. sense      percepts (+ focus, addressed, heard); others' responses to my last act learned; novelty
4. recall     the best episode and note line for the middle level (cues, hunger's recall)
5. feel       the state: situation + "A moment ago", felt phrases, intention, task, memory, knowledge
6. fast level gate signals -> salience; the slow level's act if one landed, else (flagged) head + middle level,
              else carry on
7. arbiter    bids (need, worst band, split vote, addressed, intention spent, food in sight, idle) -> maybe a thought
8. speech     `speak` without words: wait, and the urge bids for the slow level; pressure 1.0: collapse into sleep
9. act        world.act(verb) -> outcomes, effects (effectance), novelty; what the gate needs of the act
              (changed? effect? new? how many times in a row nothing)
10. intention spent?  cerebellum: predict, observe, learn -> surprise (a bid, and the gate's next-tick signal)
11. remember  strength = 1/2 salience + 1/2 novelty >= write_above -> an episode
12. update    drives and dials reduce this tick's events; knobs recomputed; adrenaline tags the last episodes
13. record    the frame (and its `log` of what changed, in full text); the distress guard
```

## World time, pause and the guard

One tick is one second of world time (`loop.seconds_per_tick`); `--pace` sets the wall seconds per tick (0.5 for
live runs: the slow level's seconds are then ticks of the mind's life). The whole world can be **paused** (the
observer, or the distress guard when a drive has stayed in its worst felt band for 90% of the last 600 awake
ticks); while paused nothing ticks and the mind does not experience it. The guard never relieves the need itself:
that would be an intervention.

## Worlds

```python
class World:
    name: str
    verbs: dict[str, str]                        # verb -> description, read by the middle and slow levels
    def sense(self) -> list[Event]               # percepts (focus, focus_by_verb, addressed ...)
    def act(self, verb, words=None) -> list[Event]
```

The brain adds its internal verbs (`rest`, `sleep`). Another mind is part of the world: others are percepts
("someone with an amber mark is here", "... said: ..."), `contact` events, and the verb `speak`, whose words only
the slow level gives. **Lands** (`land.py`): a grid defined by data (tiles, items, growth, creatures, rain, fog,
shared bodies, earshot, joint acts), whose outcomes say whether an act changed anything. **Trials** (`trials.py`):
small rooms drawn from a land's own rules, one skill each, a fresh room each waking period; `forage` is the smoke
bench. **Together** (`together.py`): several minds on one land and one clock. When bodies block (`bodies_block`),
the one in the way is told (*"Someone with a <colour> mark bumped into me, trying to go <dir>."*, 3n I1); where they
do not (the island), one squeezes past the other and both are told. The island (P3) is a
new land: a small persistent place for 4-6 minds, with seasons, water, building, giving, striking (pain, never
death), rare events and zones behind a fog. (`worlds.py`'s valley and toy garden, and the archive of origins, belong
to the earlier architecture.)

## Logs

Each run writes one JSON line per tick and mind (`runs/<name>.<bundle>.jsonl`). The meta row carries the genome's
declarations (`Brain.declarations`: each drive with its bands and phrases, each dial and knob, the verbs, the
levels' backends, the gate's weights, the head's mode, the arbiter's weights), so a new drive or hormone shows up in
the inspector without touching the page. Each frame is self-contained: percepts, act, drives, dials, knobs,
`fast` (flag, why, who chose), `probs` (the middle level's odds), `prediction`, the thought that landed; and `log`,
what changed this tick in full text: `episode`, `recall`, `thought_asked` (the prompt as rendered), `thought_cut`,
`notes`, `review`, and the night's `cues` and `head_night`. Observation keeps three things apart: observable
behaviour, what a mind declares (the slow level's words), and our interpretation (in reports only, marked as such).

## The bundle

```
bundles/<name>/
  brain.json      the genome: identity, loop settings, knobs, the levels' blocks (fast, middle, slow, attention),
                  the prompts; an individual's temperament drawn at birth
  drives.json     drive declarations
  dials.json      dial declarations
  state.json      tick, drive and dial levels, intention, task, familiarity, the head's trust window ...
  episodes.jsonl  hippocampus: {id, t, text, strength, feeling, goal, kind, consolidated, tag}
  predictor.json  the cerebellum
  wiki/           notes (front matter: title, cues, tests) + index.md; retired/
  habits/         the fast head's weights, today's labels, the held-out probe, older nights
```

A new bundle is copied from `brain/template/`. The world is not in the bundle: a resumed mind can wake in another
world. Runs: `python -m brain.trials --trial forage --bundle bundles/<a> --bundle bundles/<b> --fast laya --middle
qwen --slow qwen --pace 0.5` (the middle and slow levels need a vLLM server at `middle.url` / `slow.url`: the
middle level reads the letters' log-probabilities; `middle.lease` names an optional GPU lease, see `gpu_lease.py`);
`--fast none --middle stub --slow stub` runs locally with no model (with an overlay
`{"brain.json": {"loop": {"recall_index": {"model": "hash", "device": "cpu"}}}}` where there is no torch).

## Curiosity checklist

Written before any run, so we don't read curiosity into the logs afterwards. There is **no curiosity drive**: only
boredom, novelty, effectance and the good feeling of falling prediction error. Curiosity counts as emerged when a run
shows:

1. **Information-seeking without payoff:** acts whose only effect is new information (`look`, going where one has
   never been), while no drive other than boredom is urgent, against the same share when one is.
2. **Self-generated questions:** in the slow level's reasoning text and intentions, on thoughts no urgent
   non-boredom need asked for; counted and **read** ("what should I do next?" is deliberation, not curiosity).
3. **Exploring the learning frontier:** attention goes where prediction error is *falling*, not where the world is
   merely random (the noisy-TV trap): each place's share of awake time against its share of learning progress.
4. **Questions about itself and its origins**, read by hand in thoughts and notes.

Measured so far (earlier architecture): 0 questions in ~300 texts of the post-trained model, at ordinary
and surprising moments; one glimmer. On the island, exploration is also watched as behaviour (next phase): new
places visited per day with no reward that day.

## Non-goals

Not a model of neurons, not a claim about consciousness, not a chatbot. It is a **working architecture borrowed from
the brain's division of labour**, built to see what behaviour emerges from it. A behaviour is never concluded to be
a real emotion.
