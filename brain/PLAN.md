# Brain — plan

*The brain: an autonomous mind packaged as a bundle (a folder), put in a world, later with others. This file gives
the standing rules, the target and the phases that build it. The design as built is `BRAIN.md`. This public copy
is published one phase behind the work: P1 to P3 are done and described here; what comes next is named only.
The earlier architecture (Laya as System 1, an LLM as System 2) is v0.1 of this repository.*

## Standing rules

- **The brain is a bundle.** A folder: genome (drives, dials, knobs, prompts), episode log, wiki, state. Shutdown is
  anaesthesia; the same self resumes from the folder. No death, ever (a body can be weakened, not killed).
- **Everything is an event.** Every part publishes to one stream; drives and dials are reducers over it, declared in
  the genome: adding a drive or a hormone is configuration, not code.
- **A world is an adapter:** `sense() -> events`, `act(verb) -> events`. Another mind is part of the world.
- **Nothing of the world in the brain's code or prompts;** no concept programmed: not curiosity, love, friendship,
  empathy, morals, cooperation. They may emerge from interactions and their consequences, or not.
- **Hormones are modelled on their known human effects** (gains in the genome).
- **Learn, don't code skills.** When a mind fails at a skill, change the conditions of learning (time, a gentler
  start, feedback, a path), not the behaviour.
- **Observation keeps three things apart:** observable behaviour, the internal representation a mind declares,
  and psychological interpretation (only by us, in reports, marked as such). A behaviour is never concluded to be
  a real emotion.
- **Every run's gate is written before the run;** a result is recorded honestly, null or not.
- Innate world knowledge brought by the LLM is not a problem in itself.

## The target

### Three levels of control
Human parallels: LeDoux's low and high roads, Norman & Shallice (routines, contention scheduling, supervisory
system), Rasmussen's skill / rule / knowledge, Stanovich's autonomous / algorithmic / reflective minds. Each level
is slower and richer than the one below and is asked only when the one below cannot settle the moment.

| level | who | asked | what it does | adapts by |
|---|---|---|---|---|
| **fast** | generic code + a Laya head per mind | every tick | decides whether the moment calls for a new choice (what is in front, held or carried changed; a surprise; an alarm crossing a threshold; a new intention; the last act changed nothing); otherwise carries on (repeat the act while it works, follow the intention, or wait); later takes over the choices that have become familiar | the per-mind Laya head is trained at sleep on the middle level's choices (distillation: automatisation, as ACT-R's proceduralisation) |
| **middle** | the LLM, one pass, no thinking | when the fast level flags the tick | reads the situation, feelings, intention, what the mind knows; chooses the act (the probability of each option's letter at the first token) | through what it reads (notes, cues, intention), not its weights |
| **slow** | the same LLM, thinking | when the arbiter gives it the floor (surprise, alarm, addressed, intention spent, conflict, frustration, a need's relief in view) | deliberates: intention and task, speech; at sleep: consolidation into the wiki, review of notes, cues | through its notes and cues |

### Models and the card
- **LLM (middle and slow):** Qwen3.8-27B (an uncensored FP8 build) on vLLM, text only (`--language-model-only`).
  The middle level needs the letters' log-probabilities, so the server must return them.
- **Laya (the fast level's head):** `english` checkpoint, fp16 on the card beside vLLM (1.4 GiB).
- **One 48 GiB card holds both:** vLLM at 0.85 of the card (weights 27.6 GiB, a context cache of ~75-86k tokens
  shared by all requests in flight), Laya and the head's training at sleep in what is left.
- **The real limit is throughput:** the hybrid (Gated DeltaNet) model has no prefix caching in vLLM, so every
  middle-level call re-reads its whole prompt (~12 calls/s at most). The fast level's gate is what makes several
  minds fit on one card.

### The bundle
Drives (body needs: hunger, sleep pressure, cold, boredom, dissonance, frustration; thirst and pain on the island);
dials (hormones: dopamine, serotonin, noradrenaline, acetylcholine, orexin, cortisol, opioids, adrenaline, oxytocin,
ghrelin, melatonin) and the knobs they move; the arbiter (salience network: bids for the slow level, interruption,
rest after a thought); intention and task; working memory; episodes (written by strength, tagged by adrenaline and
oxytocin); the wiki (notes as hypotheses, reviewed each sleep); cues written at sleep with the act named from the
mind's own episodes, and a recall index rebuilt each sleep; traces written in the moment when an act relieves or
worsens a need; the cerebellum (a predictor per action and context: surprise, learning progress); speech through
the slow level; sleep (consolidation, review, cues, the fast head's distillation), entrained to the world's day.

### The world: a small persistent island
A few minds (4-6) on one clock, never named by the world (each is "someone with a <colour> mark"). Geography,
resources spread and sometimes limited so that no spot has everything (water, food, wood, stone), regrowth,
weather, seasons, rare events (storm, a wreck washing up, a spring drying, a creature), zones behind a fog, objects
whose value is not obvious. Local perception only; learning by experience or by being told. Acts: move, look,
take, use, eat, drink, speak (free words), give (to the one in front), build, work together, strike (pain and
weakness, never death). The world logs one row per round (every body, weather, season, events, changed tiles).

## Phases

- **P1 — the three-level genome (done, 2026-09-30).** The middle level (`QwenFast`: one pass, the letters' odds,
  ~0.1 s a call); the fast level (the gate's generic signals and the carry-on rule); the per-mind Laya head,
  distilled each night from the middle level's choices (observe mode first); a new template; logs that carry the
  genome's declarations; the slow level's replies shaped by vLLM's JSON schema per job. Closed by a smoke with two
  minds that passed every cell of its gate.
- **P2 — the documentation (done, 2026-09-30).** `BRAIN.md` rewritten from the code; the docs pages
  (`docs/brain-schemas.html`, how it works; `docs/brain-atlas.html`, the genome) generated from the spec and the
  template.
- **P3 — the island (done, 2026-10-01).** Built in slices, each tested alone: geography (`deploy/probe/make_island.py`
  writes `brain/lands/island.json`), the day aligned on the brain's and melatonin, water and thirst, give, strike and
  pain, build, rare events, objects of unclear value, the world's row per round. Then fixed run after run, each fix
  sorted as innate, legibility of the world, or a skill to learn: a slower body, traces written in the moment, load
  as a body sense, dozing, orexin on the clock, cortisol's feedback, entrainment to the day, sleepiness felt;
  waking spots within reach of each other and shared springs; squeezing past; frustration, the one in the way told,
  what relieves a need noticed, the effort of each thought; an intention that outlives its moment while its motive
  holds. Closed by its gate on four newborns over six island days: thoughts fast enough, nights slept, minds fed
  and watered, minds meeting. Overlays: `brain/variants/island.json` (thirst and pain on) and `traces.json`.
- **Next (not public yet):** watching the island and an inspector of the bundle; long runs.
