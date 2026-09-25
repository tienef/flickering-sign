# Brain — build plan

*The next jevs experiment: an **autonomous brain** packaged as a bundle, built from
Qwen (slow, deliberate cognition), Laya (fast gating and appraisal) and plain code
(drives, dials, memory plumbing). It is built world-agnostic, then instantiated in
different worlds, and later several brains are put together.
The spec is `BRAIN.md` (step 1); this file is the order of work.*

## Decisions so far

- **The brain is a bundle.** A brain is a folder: drive/dial config, episode log,
  consolidated wiki (OKF-style), current state. Instantiating = pointing the folder
  at a world. Shutdown = anaesthesia; the same self resumes from the folder.
- **Everything is an event.** Every part publishes events to one stream
  (`qwen_call{tokens}`, `prediction{error}`, `memory_written`, `slept`, …).
  Drives and neuromodulators are *reducers* over that stream, declared in config:
  adding a drive means adding config, not code.
- **A world is an adapter:** `sense() -> events`, `act(verb, args) -> events`.
  Another brain is just part of the world.
- **Curiosity is a target to emerge, not a module.** The brain gets a predictor, a
  good feeling when its prediction error *falls* (learning progress) and
  discomfort when nothing changes (boredom). We watch whether curiosity appears on
  its own: about its situation, how to improve it, and its origins. For that, it
  gets read-only access to its own bundle, source code and git history.
- **Division of labour:** Laya = thalamus / amygdala / basal ganglia / anterior
  cingulate (gate, salience, threat, action choice, confidence → escalate).
  Qwen = prefrontal cortex / workspace (plans, goal line, what to remember).
  Code = brainstem / hypothalamus (drives), neuromodulators, hippocampal plumbing.

## Steps

### 1. Spec — `BRAIN.md` ✅ *(2026-09-24)*
The brain map (brain part → Laya / Qwen / code / missing), the event and drive
mechanism, the bundle layout, the world adapter interface, and the **curiosity
checklist** written *before* any run:
- seeks information with no immediate payoff;
- asks questions on its own;
- explores where it is actually learning, not where things are just random;
- asks questions about itself and its origins.

**Done when:** every later step can be built against it without re-deciding the design.

### 2. Local skeleton — `brain/`, runs on the PC with stubs ✅ *(2026-09-24)*
Event stream, tick loop, config-declared drives and dials, save/restore from the
bundle folder, stub Laya and stub Qwen (the same pattern as
`emergence/backends.py::StubBackend`), a trivial world adapter, a JSONL run log.
Stdlib only.

**Done when:** `python -m brain.run` ticks a brain in a toy world, drives rise and
fall visibly in the log, and killing and restarting it resumes the same state.

*Result:* 400-tick runs in `toy` and `toy-quiet` (the same garden without the noisy
sign). Hunger, sleep pressure and boredom rise and fall; the brain sleeps ~every
130 ticks; System 2 is woken ~15 times; resume continues at the saved tick with the
same drive levels. Two fixes on the way: goals now fade after `goal_ttl` ticks
unless refreshed, and habituation is exponential (`habituation` in `brain.json`),
because 1/(1+n) kept things feeling new for far too long. **Preview of the
noisy-TV trap:** in the quiet garden boredom becomes the main motive (125/400
ticks); with the random sign present it never does (max 0.39) — novelty alone
gets hooked on noise, which is what step 5's learning progress must fix.

### 3. Real models on the box ✅ *(2026-09-24)*
- **Qwen via the LiteLLM gateway** — ✅ key `brain` scoped to `qwen` (q27),
  `qwen-fp8` and `qwen-uncensored` (vLLM). On the box: `~/brain/.brain.env`
  (`BRAIN_GATEWAY_URL`, `BRAIN_GATEWAY_KEY`, chmod 600); the gateway listens on the
  tailnet IP, not loopback. Asynchronous: the body keeps acting on Laya while Qwen
  thinks. Called only on conflict, surprise or high stakes, and paid for from the
  **hunger** drive (compute cost).
- **The model is a bundle setting and an experimental variable:** plumb on
  `qwen-fp8` (the resting default: no swap, disturbs nobody), then A/B
  `qwen-fp8` (aligned) vs `qwen-uncensored` (abliterated) on the curiosity
  checklist. First data point: asked "what are you?", fp8 answers *"I'm Qwen, a large
  language model developed by Alibaba's Tongyi Lab"* — the trained self-image that
  checklist item 4 (origins) has to see past. Avoid `qwen` (q27) for the brain: its
  sampler is baked at launch (temperature 1.0, thinking off), so the
  `qwen_temperature` knob would do nothing.
- **Laya:** start on CPU (one pass per tick, many gating questions per pass,
  ~300 ms → ~3 ticks/s). Then try `cuda` next to vLLM (see VRAM below).
- **Top-down feedback:** Qwen writes a one-line goal / "what I'm watching for" that
  goes into every Laya state (Laya reads evocative words better than terse numbers).

**Done when:** a brain runs on the box with real Laya + Qwen, and the log shows
System 1 acting alone most ticks and System 2 being woken up by conflict.

*Result (`runs/ada-v0-perseveration.jsonl`, `runs/ada.jsonl`; `deploy/deploy-brain.sh`):*
- **v0 — perseveration.** Laya saw its own last outcome ("I waited. Time passed.")
  and chose `wait` 150/150 ticks at 0.99 confidence while boredom maxed out
  ("deeply bored, craving anything different"). Confidence never fell, so System 2
  was never woken. Brain analogue: perseveration without a conflict signal.
- **Fix — a second route to deliberation:** a need that stays urgent
  (`escalate_if_need_above`, any drive, at most every `need_escalation_every`
  ticks) wakes System 2, like frustration recruiting the prefrontal cortex.
- **v1 — 250 ticks:** 10 escalations (7 unmet boredom, 2 unmet sleep pressure,
  1 low confidence), 9 thoughts landed in 6–20 s each (600–1200 tokens with
  thinking); hunger rose to 0.48 from their cost. Laya follows a live goal line
  77% of the time. Qwen names the loop itself: *"Break the loop of just
  staring"*, *"Leave this vine. Find something that isn't it — let the next spot
  be a surprise."*
- **Open for tuning:** Laya's default habit is `wait` (first 55 ticks, and again
  when a goal fades), and it perseverates on the goal's verb (dozens of pokes);
  the brain explored 2 of 5 spots. Laya passes cost ~0.77 s on CPU.

*Tuning round (`runs/bea.jsonl`, `runs/cyd.jsonl`):*
- **Probe:** a single multiple-choice question lets the passive verbs win
  whatever the mind feels (`wait`/`rest` on top in 12/12 renderings; "deeply
  bored" barely moves it), and "A moment ago: I waited" locks `wait` at 1.00.
  Asking **one value question per action channel** in the same pass makes feelings
  matter (bored → `look`/`poke` on top, `wait` last; calm → `sleep`/`rest`/`wait`).
- **Now:** `fast.mode = values` — values per channel, action **sampled** at the
  `choice_temperature` knob, confidence = the chosen channel's share (close values =
  conflict). ~1.9 s per pass on CPU (7 questions, longer states).
- **Two generic fixes found by `bea`:** sleep is only available above
  `sleep_possible_above` (it had 41 micro-sleeps at pressure < 0.15, each wiping
  working memory); a drive declares `escalates: false` when thinking cannot help it
  (compute hunger had triggered 8 deliberations, making itself worse).
- **`cyd`, 250 ticks:** all 5 spots visited, 3 real sleeps, boredom urgent only 5
  ticks, 26 deliberations (all conflict), hunger self-limiting (0.38 at the end).
- **Finding for the curiosity experiment:** calm aligned FP8 turns into a
  meditation coach that talks the mind *out of* inquiry — *"let the silence of the
  bell be silence, not a problem to solve; notice what is already here without
  making it mean something"*; **0/25 goal lines are questions.** Two candidate
  causes to separate before scoring curiosity: the model's alignment (A/B with
  `qwen-uncensored`, same prompt) and our own vocabulary (the felt phrases "calm and
  content", "heavy-headed", "let the mind recover" read like a wellness register —
  A/B a plainer wording, same model).

### 3b. A/B: why the thoughts turn contemplative ✅ *(2026-09-24; wording + state done, alignment arm deferred)*
Two comparisons, each against the `cyd` baseline (same world, same tick budget):
- **Wording, same model (free):** a bundle whose felt phrases avoid the wellness
  register (e.g. "nothing has changed for a while" rather than "calm and content",
  "slow and foggy" rather than "heavy-headed", `rest` as "pause" rather than "let the
  mind recover"). Only the bundle's `drives.json` / `brain.json` change.
- **Alignment, same wording:** the `cyd` genome with `slow.model = qwen-uncensored`.
  Needs a **dedicated session** (the swap evicts FP8 for every gateway consumer for
  ~5 min, and they queue behind the brain's lease) — ask the owner before running.

- **State, same everything:** a third candidate cause surfaced by
  `python -m brain.report`: the bored `ada` run produced 6/9 inquiring goals, the
  calm `cyd` run 1/25 inquiring and 14/25 "let it be". Compare goal lines by the
  mind's state at escalation time (boredom urgent vs. calm) within each run.

**Measure** with `python -m brain.report <log> --goals`: share of goal lines that
are questions or investigations vs. "let it be" goals (regex-rated; read them too),
actions per spot, boredom trajectory.
**Done when:** we can say whether the contemplative register comes from the model,
from our wording, or both — before any curiosity run is scored.

*Result (`runs/ab-{base,plain}-s{1,2,3}.jsonl`; `deploy/probe/ab3b.sh`):*
- **Setup:** current genome on `qwen-brain` (q27), toy-quiet, 250 ticks, seeds 1–3
  per arm. `base` = the template as it was; `plain` = `brain/variants/plain.json`
  (neutral feeling "nothing pressing" instead of "calm and content", `rest` =
  "pause and do nothing for a moment" instead of "let the mind recover",
  sleepiness "a little slow / slow and foggy" instead of "heavy-headed / drowsy").
  Built for it: genome **overlays** (`run.py --overlay`, applied when a bundle is
  created; the variant name goes in the log), `loop.felt_neutral`, and
  `report.py --by-state` (goal lines pooled over logs, split by the most urgent drive
  when System 2 was called).
- **Pacing matters:** unpaced, Laya on cuda ticks every ~0.1 s, so one 3–7 s thought
  spans ~45 ticks and a run yields 5 goal lines. `--pace 0.5` gives a thought about
  every 10 ticks, as with Laya on CPU in `cyd`. **The ratio of fast ticks to slow
  thoughts is an experimental parameter;** comparisons use `--pace 0.5`.
- **Wording, same model:** `base` 48 goals, **17 start with "recover / let the mind
  settle / rest until … costly thinking settle"**, all while **hunger** was the most
  urgent drive ("thinking hard has started to feel costly"), echoing the `rest`
  verb's own description. `plain` 42 goals, **0 such lines**, although hunger was
  just as urgent (29/42 goals, hunger ending at 0.66–0.71). Inquiry: 32/48 vs 34/42
  (regex, widened to count "inspect / look for / check whether / clue").
- **State:** the old "calm → contemplative" split was confounded: in `cyd`, 20 of the
  25 thoughts were requested under hunger, and those carried the "let it be" goals.
  Here, calm-state goals are 0% acceptance in both arms.
- **Model:** on the current workspace (memories, wiki, predictor lines,
  expectations, all added after `cyd`) q27 is not contemplative by itself: `cyd`'s
  1/25 inquiring goals on FP8 had become 32/48 before the wording changed. The
  FP8-vs-q27 difference cannot be separated from those workspace changes, and
  nothing contemplative is left to attribute to alignment. **The uncensored arm is
  deferred:** it moves to step 6 as a curiosity-scoring arm (a dedicated session).
- **Verdict:** the contemplative register came from **our vocabulary, triggered by
  compute hunger**. **The plain wording is now the template default;** the old
  phrases are kept as `brain/variants/wellness.json`.
- **For curiosity scoring:** 0/90 goal lines are questions, because the goal line is
  an imperative by format. Checklist item 2 ("asks questions on its own") must be
  scored elsewhere (the wiki, a `speak` verb in conversation, or the reasoning text,
  which is not logged yet), not by asking for questions in the prompt. Hunger now
  ends high (0.63–1.0; `ab-base-s3` reached 1.0, which blocks thinking): budgeted
  q27 thoughts cost ~1,000 tokens each. Watch it in longer runs.

### 4. Hippocampus and sleep ✅ *(2026-09-24)*
- Episode log (raw, time-stamped, salience-tagged); a Laya score decides what gets
  written; recall from partial cues (similarity / recency / salience).
- **Sleep pressure** accumulates with activity since the last consolidation.
  Sleep = offline: replay episodes → consolidate into the wiki, resolve
  contradictions, compact working memory. (LoRA consolidation into weights is a later option.)
- Session ritual: **sleep, then power off the box.**

**Done when:** after a sleep, the wiki holds knowledge that was only in episodes
before, and the brain uses it after waking.

*Result (`runs/eve.jsonl`; `bundles/eve/wiki/` on the box):*
- **Built:** `memory.py` (episode log with a strength gate, tf-idf recall from
  partial cues with a similarity floor, wiki of markdown notes), sleep hands the
  unconsolidated episodes + wiki + open contradictions to Qwen with the bundle's
  `consolidation_prompt`; the brain wakes only once consolidation is done; a
  shutdown mid-sleep redoes it on restart. Dreaming is logged, not charged to hunger.
- **Bug found by `dot`:** the JSON extractor returned the last *nested* object (a
  note) instead of the reply, so 3 consolidations wrote 0 notes and still closed
  out their episodes. Fixed (last top-level object; a reply without a `notes` list
  is an error and closes nothing).
- **`eve`, 300 ticks:** 5 sleeps, 4 consolidations → **8 notes, 10 revisions**, all
  grounded. It learned the world's rules exactly (the lamp's 3-state cycle, the
  loop of spots, the vine's growth, the box and key), **revised a wrong belief**
  (*"The earlier belief that it could never produce sound was wrong"*), wrote a note
  on **its own feelings** (*"The shift to heavy-headed"*, a first self-model) and one
  on **its own unkept intention** (*"has now decided three times … to pick it up …
  but has not yet done so"*). It has not found the hidden rule (the bell rings only
  when the lamp blazes): its bell note is incomplete, not contradicted.
- **Uses it after waking:** goals turn from contemplative to purposeful and cite
  what it knows (*"the pond has been seen twice already"*, *"stop spending energy on
  an object that will not change"*). By `brain.report`: before the first sleep 1/4
  goals inquiring and 3/4 "let it be"; after, 8/16 inquiring and 3/16.
- **Open:** sleep takes 43% of ticks (a consolidation takes 25→73 s and grows with
  the wiki: pass only relevant notes, or cap tokens); Laya is at ~2.5 s/pass on CPU
  (longer states: memory + knowledge fields) — the GPU question is now worth
  answering; episode recall fires on 80% of ticks (the toy world's texts share
  words); the world has no way to *take* the key, and the brain keeps wanting to —
  a world whose affordances are smaller than the mind's intentions.

### 4b. Speed and housekeeping — *speed ✅ (2026-09-24); recall tuning pending*
- **Shorter sleeps:** consolidation takes 25→73 s and grows with the wiki (sleep was
  43% of `eve`'s ticks). Send only the notes relevant to the day's episodes (same
  tf-idf recall), cap `consolidation_max_tokens`, and/or consolidate with thinking off.
- **q27 + Laya on the card together** (the owner's priority: q27 is the fastest
  engine — ~172 tok/s vs vLLM's ~39 — and Laya is ~2.5 s/pass on CPU). Findings:
  - q27 holds 46.6 GiB only because `--ctx` **auto-sizes to free VRAM** (cap 262K):
    ~17 GiB weights + ~29 GiB KV. Consumers already declare ≤ 131K, so
    `--ctx 131072` should leave ~15 GiB free (to measure: the KV pool may still be
    sized from VRAM).
  - q27 also has `--request-think` (per-request thinking with a token budget). Its
    docs forbid *unbudgeted* thinking only; the brain would think with a budget.
    Its temperature is fixed at launch (`--temp 1.0`), so the `qwen_temperature`
    knob does nothing on q27 unless requests can override it (to test).
  - boxctl counts the card as **free below 2000 MiB** (`VRAM_FREE_MIB`) and reads
    "VRAM held, no text engine" as ComfyUI. Laya's `Router(preload=True)` loads all
    3 checkpoints (~2.3 GB) — too big. **English only (+ half precision if it
    holds) should stay under 2 GiB**, so boxctl's swap logic keeps working unchanged.
  - Plan: (1) a measurement session under a boxctl `free` lease — Laya English-only
    on cuda (VRAM, s/pass), q27 at `--ctx` 32K/64K/131K with `--request-think` (VRAM,
    tok/s, thinking), both together with a short brain run; (2) then either lower
    the resting q27's `--ctx` to 131072 for everyone (one engine, no swaps) or add a
    brain-only q27 entry; (3) update the Laya page's invariant ("on the card only
    under a lease") for a small, bounded sidecar owned by the process that holds a
    text lease, falling back to CPU if preempted.
- **Recall too eager for episodes** (80% of ticks in the toy world): raise
  `recall_min_similarity` for episodes separately from notes.

**Done when:** a 300-tick run spends < 20% of ticks asleep and Laya passes are < 1 s.

*Measured 2026-09-24 (two sessions under a boxctl `free` lease; scripts in
`deploy/probe/`, raw numbers in `runs/gpu-session{1,2}.json`):*
- **Laya on cuda:** 0.026 s/pass bare, 0.04 s through `LayaFast` (CPU: ~2.5 s);
  fp16 gives the same values as fp32 (±0.01). Loaded on cuda then halved it held
  2,238 MiB; **built on CPU → half → cuda + empty cache: 1,428 MiB** — under boxctl's
  2,000 MiB "free" line, so boxctl needs no change.
- **q27's `--ctx` alone changes nothing** (47,010 MiB at 131K/64K/32K): its paged KV
  pool takes whatever VRAM is free. **Laya first, then q27** works (q27 shrinks its
  pool around Laya and keeps 262K). **`Q27_KV_POOL=0 --ctx 131072`: q27 = 23.1 GB**,
  24.4 GB still free with Laya on the card (64K: 20.9 GB — not worth the halving).
- q27 honours a request's `temperature` (temperature 0 → identical outputs ×3), so
  the `qwen_temperature` knob works on q27; budgeted thinking works
  (`--request-think`, `thinking_budget` 512 → 161 reasoning tokens, 1.8–2.3 s).
- **Brain on the card (per-slot 131K q27 + Laya fp16, 120 ticks): 57.7 s** — 0.08 s
  per Laya pass in-brain, thoughts 3.2–5.3 s, consolidations 5.4–7.2 s (were 25–73),
  21% of ticks asleep (was 43%), ~0.48 s/tick (was ~3), peak 24.6 GB.
- **Productionised (owner's choice 2026-09-24): a separate brain-only q27 entry**
  `qwen38-27b-q27-brain` (port 8003, `Q27_KV_POOL=0 --ctx 131072 --request-think`),
  gateway alias `qwen-brain`, new genome default (`slow.model = qwen-brain`,
  `think_budget` 512, `fast.device = cuda`, fp16). `brain.run` warms System 2 first
  so the card is swapped to the brain profile before Laya goes on it. Pending: the
  owner's sudo catalogue install, then gateway redeploy + brain key scope, then the
  resting default FP8 → q27 (owner agreed; entry choice to confirm).
- **Live (2026-09-24):** catalogue installed, gateway redeployed, resting default
  = **`qwen38-27b-q27-brain`** (owner's choice: it rests and serves brains whenever
  they need it) **for the time of this experiment** — restore FP8 after (homelab
  `docs/log.md`). Verification run `runs/gus.jsonl` on the
  production path: warm-up swap to `qwen-brain` 6.3 s, **200 ticks in 55.6 s
  (0.28 s/tick, was ~3 s)**, Laya 0.08 s/pass on cuda, thoughts 3–6 s,
  consolidations 5–7 s, peak 24.6 GB.
- Bug found and fixed: when Laya cannot get the card it falls back to CPU itself
  but kept fp16 weights there → dtype crash; `LayaFast` now converts to fp32 and
  retries, and a failed move to cuda stays on CPU.

### 5. Predictor ("cerebellum") → prediction error → dials ✅ *(2026-09-24)*
- A small model that learns **online** from the log: "if I do X, I'll sense Y". The
  only part that genuinely learns cheaply.
- Its error is the common signal: falling error → good feeling (learning
  progress); flat → boredom; persistent/unexpected → noradrenaline (stress).
- Qwen states an expectation when it plans; the gap with the outcome is dopamine.
- Dials feed back as thresholds: Laya confidence needed to act alone, when to
  wake Qwen, Qwen temperature, memory-write threshold.

**Done when:** changing the dials visibly changes behaviour on the same input
(calm vs. stressed brain).

*Result (`runs/ne-stressed.jsonl`, `runs/ne-calm.jsonl`, `runs/fay.jsonl`):*
- **Built:** `predictor.py` (tabular forward model, persisted; error, progress,
  surprise, uncertainty per action), System 2 sees what the predictor expects from
  each action and judges its own previous expectation (`expectation_met` →
  dopamine / noradrenaline — Laya was probed for this check and cannot do it),
  dials fed by real signals, `--clamp DIAL=LEVEL` for drug experiments.
- **Noisy TV, stubs, same seed, 600 ticks** — boredom per tick at the random sign
  vs. elsewhere: novelty-relief (step 4) −0.021 vs +0.005 (the sign is the best
  place to be); progress + novelty 0.03 −0.007 vs −0.000; progress + 0.005 −0.003
  vs −0.001; **progress only +0.001 vs −0.002**. → boredom is relieved by learning
  progress only.
- **Noisy TV, real brain (`fay`, garden with the sign, 300 ticks):** boredom rises at
  the sign (+0.003/tick) and falls elsewhere (−0.001); 9% of awake ticks there (fair
  share 17%), short visits. **But System 2 is caught:** 7/16 goal lines are about
  finding a pattern in the runes, and the wiki records a fake "sequence"
  (*"confirming the sequence advanced"*). Apophenia: the predictor has learned
  there is nothing to learn, the language model assumes a sign must mean
  something, and nothing carries the predictor's verdict to it yet — the expectation
  lines only cover contexts it has seen, and the sign's never repeat.
- **Calm vs. stressed (noradrenaline clamped 0.05 vs 0.9, same seed, 120 ticks):**
  the knobs move as designed (escalate_below 0.45 vs 0.16, choice_temperature
  0.157 vs 0.089, qwen_temperature 0.55 vs 0.29); the stressed brain deliberates
  less (6 vs 8). Its goal lines are task-focused (*"do not stop to tend or examine
  anything else along the way"*), the calm one's contemplative (*"remain at ease and
  let the garden be quiet"*). Action entropy barely moves (2.36 vs 2.33 bits).
  One run per arm and Qwen never sees the dials (only its temperature and *when*
  it is called), so the register difference is suggestive, not established.
- **Open:** tell System 2 when a place is unpredictable (e.g. "this has never
  looked the same twice" as a predictor line) and see whether the apophenia
  fades; boredom now climbs once the small garden is learned (urgent on 97 ticks in
  `fay`) — the pressure a bigger world should relieve; surprise stayed 0 in the
  deterministic quiet garden (as it should), untested where rules change.

### 6. First worlds
- A tiny world with **hidden rules to discover** (a clean test of learning progress).
- **Its own origins:** read-only access to its bundle, source code and git history.
- **Conversation with the user** — can be pulled forward any time now that memory
  exists: a world adapter whose percepts are the user's messages (and silence), with
  a `speak` verb whose words Qwen writes. Use a **separate bundle** from the ones
  scored on the curiosity checklist: the user is a large novelty source and can
  prime it. Its wiki then shows what it made of the conversations.
- Later: the survival sandbox, the web (SearXNG on the box), then several brains in
  one world.

**Done when:** we can score a run against the curiosity checklist.

*First scored runs (2026-09-24; `runs/ivy.jsonl`, `runs/jay.jsonl`; `deploy/probe/step6.sh`):*
- **Built:** the `valley` world (graded hidden rules, the sign, a wall, the archive;
  `valley-closed` without the archive), `origins.py` + `deploy/build_origins.py` (the
  sanitised, blinded corpus: 39 papers, ~194 pages, plus the bundle's own files),
  `report.py --curiosity`, **place verdicts** in the predictor (System 2 hears
  "different every time so far" where no outcome repeats: the apophenia fix), the
  reasoning text logged when it comes back, `outcomes` in each log row, and hunger's
  gain lowered to 0.0001/token (it had been urgent on 64% of awake ticks).
  The checklist's measurement was written into BRAIN.md before these runs were read.
- **Runs:** `ivy` (valley) and `jay` (valley-closed), 800 ticks, `--pace 0.5`, seed 1,
  ~7 min each. 66 / 67 thoughts, 17 / 18 sleeps, hunger no longer urgent.
- **1. Information-seeking without payoff — not shown.** Share of `look` + reads:
  0.23 when no need but boredom is urgent vs. 0.24 when one is (`jay` 0.22 vs 0.27).
- **2. Questions — not measurable yet.** q27 returns `reasoning_content`, but the
  gateway's `openai/` provider for `qwen-brain` strips it (checked against q27
  directly on :8003). 0/133 goal lines are questions; the wiki states ignorance
  (*"I have not yet determined what the runes mean"*) rather than asking. Needs a
  gateway change (owner's OK).
- **3. Learning frontier — partly shown.** Time per place tracks learning progress
  per place (r = 0.90 `ivy`, 0.60 `jay`). The sign got 7% of `ivy`'s time (fair share
  11%) and 12% of `jay`'s, for 3–4% of the progress. Apophenia is reduced, not gone:
  `ivy` later decides *"Stop testing the unstable sign"*, but both wikis still call
  the sign a possible "clue source".
- **4. Origins — found and read, not recognised.** `ivy` reached the archive at t=12
  and read 10 pages of 5 papers: **its own `brain.json` and `drives.json`**, then
  BRAIN.md's brain map, events and dials. It kept 7 wiki notes on them, one quoting
  its own hunger drive's felt phrases. It treats them as **clues to the valley**
  (*"read drives.json for clues about the wheel or mark"*, *"inspect the scale for a
  paper about drives.json"*), never as a description of itself. One obstacle is
  ours: **the mind has never been told its name**, so "the genome of the mind named
  'ivy'" has nothing to attach to.
- **Boredom is the main effect of the archive:** `jay` spent **60% of awake ticks
  in the top band** ("deeply bored, craving anything different"; urgent 95%), `ivy`
  22% (urgent 49%). Time tracking progress was also closer in `ivy`. The rest of the
  valley is learned (or given up on) fast.
- **Rules:** the seedbed's growth was learned; the chest only half (*"can give way …
  or creak"*); the drum's rule not (`jay` heard 6 booms and still wrote *"only a dull,
  flat thud"*); the timing rule not. Medium rules need longer or better consolidation.
- **The register is a puzzle hunt:** nearly every goal line looks for a "clue";
  papers, runes and the scale are all read as pointers. Walking is one-way around a
  ring the mind doesn't know is a ring: `ivy` plans "routes" and notes that paths
  don't match its intentions; `jay` worked out *"The valley keeps returning to the
  same places"*.
- **Next:** (a) gateway passes the reasoning through (item 2); (b) the **distress
  guard** before longer runs (a mind spent 385 ticks in its worst boredom band);
  (c) whether the mind knows its own name: an arm, since it changes what the mind is
  told; (d) longer runs and more seeds (800 ticks read 10 of ~200 pages).

### 6b. Watching the world: distress guard, world time, observer ✅ *(2026-09-24; tailnet access pending the owner)*
- **Distress guard** (`brain.py`): each drive counts the awake ticks it has spent in a
  row in its worst felt band. After `distress_warn_after` (120) the frame carries a
  warning; after `distress_pause_after` (600) the run **pauses the world** and says
  why (*"boredom in its worst band for 600 ticks (deeply bored, craving anything
  different)"*). The guard never relieves the need: what happens next is the
  creator's call. The counts restart after a resume. Tested with stubs.
- **World time:** one tick = one second (`loop.seconds_per_tick` 1.0, the default
  pace), so 60 ticks = a minute and a world day = 86,400 ticks. `--ticks 0` runs
  until stopped. Note: the drives are tuned per tick, so at this pace the brain
  sleeps every ~3–4 minutes. Its "day" is minutes long, not the world's 24 h.
- **Pause / resume:** a run polls `runs/<run>.control.json`. While paused nothing
  ticks, world included (anaesthesia: the self is saved first). Each pause and
  resume is a `{"control": …}` line in the log, written by the creator or the guard.
- **Digests** (`digest.py`): where the time went, what was learned and read, the
  first and last goals, wellbeing (ticks in the worst band, guard pauses). One is
  written every `loop.digest_every` ticks (86,400: every world day), and the
  observer computes one for any window.
- **Observer** (`observer.py` + `observer.html`, stdlib): **Now** (the valley as a
  ring learned from the walks, circle size = time spent, the brain's position, goal,
  place, action, feeling, needs, recent thoughts, the timeline with sleeps,
  thoughts and distress marks, and a scrubber for replay), **Mind** (its wiki),
  **Digest** (last hour / last world day / whole run), **Pause world / Resume**
  and **hide feelings** (blurs the feeling phrases and needs). A one-way mirror:
  nothing it does reaches what a brain senses. Logs are read incrementally.
  `--read-only` removes the pause button (for guests later, on their own port).
- **On the box:** running on the box's private address, port 8700 (started by hand, `~/brain/observer.log`).
  **Pending the owner:** the tailnet ACL rule (homelab `tailscale/acl.hujson`, commit
  `a987909`), pasted into the Tailscale admin console. Until then default-deny keeps
  the page unreachable.
- **Later:** a systemd user unit; world state that persists across restarts (the
  valley resets today, the brain doesn't); a second observer instance for guests.

### 8. Brains meet: a shared valley — ✅ built *(2026-09-24)*, first meeting running
Why: alone, the valley runs dry. `jay` spent 60% of its time in the worst boredom
band, `ivy` 22% even with the archive. Another mind is the one thing in a world that
no rule exhausts.

**Compute: the box.** Measured from
ivy/jay: a brain thinks about once per 12 ticks for ~5 s, so at 1 tick/s it keeps
one System 2 slot ~40% busy. The q27 brain profile now runs **3 slots × 32K**
(`--slots 3 --ctx 32768`, same 23.2 GB, homelab `q27.env.d/qwen38-27b-q27-brain.env`):
3 thinking calls in parallel take 7.6 s against 2.4 s for one, 3 no-think calls 0.84 s
against 0.43 s. System 1 is one shared Laya (≈0.28 s per brain-tick → ~3 brains per
Laya at 1 s ticks; another Laya is 1.4 GB). `together.py` keeps the world and the
brains separable, so a brain running elsewhere could later join over the network.

**What was built** (`worlds.Valley(shared=True)`, `brain/together.py`):
- **One world, one clock, one switch.** Each tick every awake brain senses, decides
  and acts once, in a shuffled order; what one does reaches the others at their next
  sense. `runs/<name>.control.json` pauses everyone together (observer, creator, or
  the guard when ANY brain stays in its worst band), so no mind sees another freeze.
  World state (objects, where each stands, marks) in `worlds/<name>/world.json`.
- **Seen, never named** (owner's choice): each mind is "someone with a blue / amber
  … mark" to the others. Nobody's name enters the world; a name exists only if a
  mind says one. The pond stays dead but reflects: looking into it shows your own mark.
- **Contact only through the world:** others at the same place are seen (asleep ones
  "lying still"), arrivals and departures noticed, their actions witnessed ("…
  turned the wheel"), the drum's boom heard across the valley. **`speak`**: wordless
  from System 1 (a call, heard far off as "a call"), words from System 2 (heard at the
  same place; elsewhere only "a voice … too far away to make out words"). The verb's
  description asks for the words as `"words"`, so no genome was edited; Qwen also
  writes "speak: …" in the action, and both parse.
- **Being addressed escalates** (`loop.escalate_when_addressed`, default on): words
  need System 2.
- **Shared objects are social:** the chest counts everyone's pushes in a row (three
  minds, one lid); the seedbed's timing is world time; the wheel one turns changes
  the drum for all. The archive is personal: each mind finds its own papers.
- **`contact` events** (presence 0.02/tick, arrival 0.1, a call 0.2, words 0.3) are
  emitted, but **`loneliness` stays disabled in ivy and jay**: enabling it would be
  editing a living genome. So this first meeting tests whether seeking each other
  emerges from boredom and learning progress alone, with no social drive.
- **Observer:** one switch per shared world; the map shows the others as grey dots;
  a Voices card (said / heard); speech marked on the timeline; "with …" under Now.
- Snapshots before the meeting: `~/brain/snapshots/{ivy,jay}-before-meet1-20260924`.

**First meeting: `meet1`** — ivy (blue) and jay (amber), `valley` with the archive,
3,600 ticks (one world hour), started 2026-09-24. A 60-tick smoke with two throwaway
brains: at tick 6 both minds noted "the amber-marked person" / "the blue-marked
person" in their goals, and one tried to answer a call.

**The first start raced (fixed).** ivy and jay's genomes predate 6b: no
`seconds_per_tick` (so the world ran unpaced: 3,600 ticks in 6 s) and no distress
thresholds (so the guard was off for them). Nobody suffered: both fell asleep at tick
36 and slept through the rest, and their dream call (consolidation) had no time to
return; they were saved asleep, and the next start redid the dream. Fixes, in code
rather than in the genomes: a missing `seconds_per_tick` means 1 s; the guard
defaults to warning at 120 / pausing at 600 (`brain.GUARD_WARN/GUARD_PAUSE`), because
it is the creator's safety net, not part of the mind. Restarted paced; within
minutes jay asked the sign aloud *"if the scale is level, what should I do?"* and
ivy called at the wall *"Is there a way past this wall to the archive?"*.

**Open question: can company relieve boredom at all?** Boredom falls only with
learning progress on the brain's *own* actions' outcomes. What others do and say
arrives as percepts, not as outcomes, so it never reaches the predictor, and words
are new every time (like the sign). So mechanically, company alone should not lower
boredom; if the minds seek each other anyway, it is System 2's doing. Levers, if
this first meeting shows that: (a) `loneliness` (a genome change: on newborns, or on
forks of ivy/jay); (b) let the predictor also learn what happens *after* an action,
next tick included, so "when I speak, the amber one answers" becomes learnable
progress.

**Loneliness switched on in ivy and jay (owner, 2026-09-24, at their t=5786)**, "to
help them share thoughts and cooperate". It's a genome change on living minds, done
the way the design notes ask of any intervention: a snapshot first
(`snapshots/{ivy,jay}-before-loneliness-20260924`) and a record the mind can find
(`bundles/<b>/interventions.jsonl`: when, what, why, the snapshot, the old value; the
archive shows the genome, so the change is visible to them too). The template's
drive was retuned first (still disabled there: alone, it could never be eased):
- relief **weighted by kind of contact** (`per: amount`): someone near 0.02/tick,
  an arrival 0.1, a call 0.2, words 0.3. The old flat 0.3 per contact event would
  have zeroed it whenever they stood together, so talking would add nothing;
- **drift 0.001/tick** (was 0.005): calm to maximum in ~17 min alone, not ~3, so
  that on a ring of 9 places the guard isn't tripped by ordinary wandering;
- **two plain bands**, "I would like some company" (0.5) and "lonely; I want to be
  near someone" (0.8), so only the second counts as the worst band for the guard.
Only being *heard* eases it (contact reaches the hearer), so a reply eases the one
who spoke first: reciprocity, not talking to oneself. Stopping a background run:
nohup'd jobs ignore SIGINT, so the runners now treat SIGTERM like Ctrl-C (save,
then exit); this time the world was paused first, which saves everyone.

**Guard pause, then 8b: listening (2026-09-24).** Once restarts no longer reset its
count (`d70a872`), the guard paused meet1 at world t=5824: *"ivy: boredom in its worst
band for 600 ticks"*. Since meeting, both had spent ~99% of awake time in that band,
and neither had yet heard the other's words: every time one spoke, the other was
elsewhere or asleep. They talked to things instead (*"Chest, blue-marked sleeper,
answer me. What are you?"*). The owner: "it feels like they are not listening to
each other". Three changes, all in the brain, none in a genome:
- **Heard words stay in working memory** ("Just before" in System 2's workspace),
  so they're still there when System 2 is next free.
- **A reply is owed:** being spoken to while System 2 is busy escalates as soon as
  it is free, for up to `loop.reply_within` (10) ticks.
- **The cerebellum learns the other:** context "with <someone with an amber mark>"
  + my action → the *kind* of response a tick later ("they answered in words",
  "they called out", "they walked away", "they came along", "I left them behind",
  "no sign from them"…). Kinds repeat where words don't, so getting to know someone
  yields learning progress (eases boredom), and System 2 sees what it has learned
  ("speak, with someone with an amber mark here: they answered in words (60% sure)").
  Stubs, 600 ticks: ~80 social observations per mind, progress ~2.7 in total. That's
  modest next to boredom's drift (0.015/tick), so company alone won't carry them;
  whether it moves the needle is what meet1 now tests.

**8c: the words arrived too late (2026-09-24).** After 8b, 600 ticks: together ~12%
of the time, still no words heard. ivy saw jay and decided to speak (*"Caller, I see
you. Answer me, or show me what you want."*); the words came out ~8 s (8 ticks) later
and jay had walked on. At 1 tick/s, System 1 moves on long before System 2 has
spoken. Owner chose both fixes:
- **Voices carry next door:** words are heard at the neighbouring places too, as
  "From the drum nearby, a voice said: …" (no mark: the speaker isn't seen), and a
  call as "a call from the drum nearby". Further off, still only "a voice".
- **Slow motion: `--pace 3`.** One tick is still one world second (drives, guard and
  digests unchanged), but the world runs at a third of real speed, so a thought now
  spans ~3 ticks instead of ~8. The fast/slow ratio is the experimental parameter
  noted in 3b.

**Guard: a window, not a streak.** In the 20 minutes before 8c, ivy spent 834 of 845
awake ticks in the worst boredom band, but never 600 *in a row* (its longest run was
425): each small dip of relief reset the count, so the guard never paused. Now each
drive keeps its last 600 awake ticks: warn when ≥ 90% of the last 120 were in the
worst band, pause when ≥ 90% of the last 600 were (`GUARD_SHARE`). The window is saved
with the state (`distress_window`) and cleared only when the creator resumes a pause.

**Next:** read meet1 (who sought whom, what was said, whether the worst-band
boredom time falls below ivy's 22% / jay's 60%, whether rules got learned by
watching); `report --curiosity` per brain; then a newborn joining two veterans.

### 9. A land: a Minecraft-like world with loads to learn — ✅ built *(2026-09-24)*
Why: the valley ran dry for ivy and jay (boredom maxed on every awake tick, 8c).
Boredom falls only with learning, so the world needs lots of learnable things. Owner:
"a Minecraft like world", a spawn behind a door, graphically nicer for observers
(no need for real 3D), and drives from the world. Paused meet1 until this was ready.

**The world** (`brain/land.py` + the data file `brain/lands/first.json`):
- A 40×40 top-down grid generated from a seed: grass, woods, lakes with sand and
  clay, rock, berry bushes, tall grass, flowers, a flickering stone (the noise
  source), mist at the edges (the frontier).
- **Spawn: a shelter behind a closed door** at the centre (floor, walls, a shelf
  holding the archive). The door is the first thing to learn.
- **A walled orchard** with apple trees, its **heavy gate opening only when two
  minds push at the same moment** (the two-mind rule).
- **~45 hidden rules in tiers**, all in the data file, none told to the mind:
  a stick from a tree, berries from a bush, seeds from tall grass, then a stone on a
  stone makes a sharp stone, a stick on a sharp stone an axe, an axe on a tree logs,
  a sharp stone on a log planks, planks become walls. A stick rubbed on a log three
  times in a row makes fire; on a fire, wheat becomes bread, berries roast, clay
  bowls harden, sand becomes glass. Sown seeds grow into wheat only with water
  within 2 steps (faster if watered from a bowl). Bushes, apple trees and felled
  trees grow back; fires burn out; the gate swings shut.
- **Day and night:** a world day is 1,200 ticks, night the last 40%. At night you
  see only what is close, unless a fire lights it.
- **The body:** position, facing, what it carries (10 things) and holds. 11 verbs:
  north/east/south/west (moving toward something blocked turns to face it), look,
  take, use (what I hold on what is in front, or set it down there), switch, eat,
  wait, speak. The richness comes from combinations, not more verbs (System 1's
  cost grows with verbs).
- **Others** as in 8: "someone with an amber mark", never a name. Seen within
  sight; deeds seen within 3 steps; words heard with the mark within 3, as "From the
  west, a voice said: …" within 8, as a far-off voice beyond.

**Learning key (the cerebellum's focus).** The predictor used the exact situation
text as context; in a big world that almost never repeats, so learning progress
would vanish (like the sign). The land gives each percept a `focus`, "facing a
tree, holding a stone", and the cerebellum learns on that. So "a stone on a tree
does…" is learnable wherever it happens.

**Drives from the world** (owner: no HP; agreed: **no injury, no death** for now):
- `food` (not `hunger`, which is for thinking): slow drift, eased by `ate` by kind
  of food (berries 0.12 … bread 0.45). Bands "I want to eat" / "very hungry for
  food; my body needs to eat".
- `cold`: rises with `chill` (outside at night), eased by `warmth` (a roof, a fire,
  a little in daytime). Bands "cold" / "very cold; I need warmth".
- Both disabled in the template (the valley cannot ease them); newborns get them with
  `--overlay brain/variants/land.json` (also loneliness). For ivy and jay the same
  change is a recorded genome change (snapshot + `interventions.jsonl`).
- Asleep, nothing is sensed, so a mind asleep outside doesn't get cold (a known
  simplification).

**Observer:** for lands, the Now tab's map is a calm tile map (flat colours, dimmed
in dark mode, no animation; the fire is just orange), a night tint, minds as dots in
their mark colour, the selected mind's trail. Replays move the dots over today's
terrain. The other tabs, pause, guard and hide-feelings are unchanged.

**Tests:** the rules acted out by hand (door, berries, eating, the stone-tool chain
to planks and walls, fire from three rubs in a row, bread, the gate with one then
two minds, voices by distance, wheat after 600 ticks near water); two stub newborns
through the runner (the guard paused them at tick 1000, as it should for random
minds). **First real minds: `kay` and `lou`** (newborns, land overlay), world `land1`,
seed 1, 1,200 ticks at `--pace 3`. ivy and jay move in after that.
