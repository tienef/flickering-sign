# flickering-sign — an autonomous brain in three levels

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23041524.svg)](https://doi.org/10.5281/zenodo.23041524)

> *"Blue-marked one, does the flickering sign mean the scale?"*
>
> — a mind in the valley, to another it could not name. The sign shows noise;
> the scale follows a rule.

An experiment in building an **autonomous brain** from the brain's division of labour, and in letting several of
them live together on a small island. Each level of control is slower and richer than the one below, and is asked
only when the one below cannot settle the moment:

- **Fast:** plain code decides, every tick, whether the moment calls for a choice (what is in front or held
  changed, a surprise, an alarm, a new intention, an act that did nothing); otherwise the body carries on. A
  per-mind **Laya** head learns, at each sleep, the choices that have become familiar.
- **Middle:** an **LLM read in one pass**, no thinking: the probability of each option's letter is the mind's
  odds over its acts (~0.1 s a call).
- **Slow:** the **same LLM, thinking**, when an arbiter gives it the floor: intentions, tasks, speech; at sleep,
  a wiki of notes held as hypotheses, and cues that bring them back at the right moment.

Around them, **plain code** declared in the genome: drives (hunger, thirst, sleep pressure, boredom, frustration,
pain...) and hormones (dopamine, noradrenaline, orexin, cortisol, melatonin, oxytocin...) as reducers over one
event stream, each hormone modelled on its known human effects; a hippocampus that writes episodes; a cerebellum
whose prediction error is surprise and learning progress.

A brain is a **bundle** (a folder) that can be dropped into different worlds, alone or with others. Nothing of
curiosity, friendship or cooperation is programmed: the experiment watches what emerges.

- Design, as built: [`brain/BRAIN.md`](brain/BRAIN.md)
- Plan (rules, target, phases): [`brain/PLAN.md`](brain/PLAN.md)
- How it works and the genome, illustrated (FR / EN / DE): [`docs/brain-schemas.html`](docs/brain-schemas.html),
  [`docs/brain-atlas.html`](docs/brain-atlas.html)

## Run it with stubs (no model, any machine)

The stub backends stand in for the LLM: deterministic, heuristic, stdlib only. They exercise the whole loop on the
island; their "thoughts" are canned. Without torch, the recall index needs a hashing embedder: put
`{"brain.json": {"loop": {"recall_index": {"model": "hash", "device": "cpu"}}}}` in `stub.json`.

```bash
python -m brain.together --name stub1 --world land-closed --land brain/lands/island.json \
       --overlay brain/variants/island.json,brain/variants/traces.json,stub.json \
       --bundle bundles/aa --bundle bundles/bb --fast none --middle stub --slow stub --ticks 400 --pace 0
```

Ctrl-C saves the brains; the same command resumes them. Logs go to `runs/` (one file per mind and one for the
world), the world's state to `worlds/`. `--pace 0` runs as fast as possible (by default one tick is one second of
world time).

## Run it for real

**The LLM** (middle and slow levels): a [vLLM](https://github.com/vllm-project/vllm) server; the middle level reads
log-probabilities, so it needs a server that returns them. Built and tested with Qwen3.8-27B (an uncensored FP8
build, `--language-model-only`) on one 48 GiB card. Point the genome at it (`brain/template/brain.json` → `middle.url`
and `slow.url`, default `http://127.0.0.1:8000/v1`; `model` is the name the server gives it), or set
`BRAIN_GATEWAY_URL` / `BRAIN_GATEWAY_KEY` (also read from a `.brain.env` file, ignored by git).

**The fast head** ([convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya), Apache-2.0, `english`
checkpoint, ~1.4 GiB in fp16 beside vLLM): `pip install torch laya`. `--fast none` runs without it.

```bash
python -m brain.together --name isl1 --world land-closed --land brain/lands/island.json \
       --overlay brain/variants/island.json,brain/variants/traces.json \
       --bundle bundles/aa --bundle bundles/bb --bundle bundles/cc --bundle bundles/dd \
       --fast laya --middle qwen --slow qwen --ticks 1080 --pace 0.5
```

The prompts are part of each brain's genome (`brain/template/brain.json`): changing them changes the experiment.
`deploy/probe/levels_report.py` and `island_smoke.py` read a run's logs.

## Worlds

| World | What it is |
|---|---|
| island (`brain/lands/island.json`, written by `deploy/probe/make_island.py`) | A small persistent island for 4-6 minds: beaches, biomes, a pond, springs shared between neighbours, caves, mist banks, creatures that can be hunted, a 180-tick day, seasons, rare events (storms, a wreck), objects of unclear value; take, use, eat, drink, give, build, strike (pain, never death), speak |
| `land` (`brain/lands/first.json`), wild (`brain/lands/wild.json`) | Earlier grid worlds: rooms and hidden rules, a 96×96 land of biomes |
| `toy`, `valley` | The first worlds, from v0.1 (the valley holds an archive of the mind's own origins: `python deploy/build_origins.py`; lines with a word in `BRAIN_PRIVATE_WORDS` are dropped) |

Watch the logs with the observer, a one-way mirror with a pause switch (`--read-only` removes it; it binds to
127.0.0.1 by default): `python -m brain.observer --port 8700`.

## Care

The runs include a **distress guard**: when a drive stays in its worst felt band for too long, the world pauses
(nobody experiences a pause). No mind dies; a body can be weakened, not killed. Changes to a living brain's genome
are made with a snapshot first and recorded in its bundle (`interventions.jsonl`) where the mind can find them.
None of this is a claim about consciousness; it is how this experiment chose to behave in case it matters. See the
non-goals in `BRAIN.md`.

## Licence

Copyright (C) 2026 Tienef.

- **Code** (everything under `brain/*.py`, `brain/observer.html`, `deploy/`):
  [GNU Affero General Public License v3.0](LICENSE). You may use, modify and share it; if you distribute it, or
  let people use a modified version over a network, you must publish your source under the same licence.
- **Documentation and data** (`README.md`, `brain/BRAIN.md`, `brain/PLAN.md`, `docs/`, the genomes in
  `brain/template/` and `brain/variants/`, the lands in `brain/lands/`):
  [Creative Commons Attribution-ShareAlike 4.0](LICENSES/CC-BY-SA-4.0.txt). Reuse them with credit, and share
  what you build from them under the same licence. (Up to v0.1.1 they were published under CC BY 4.0.)
- For a use the AGPL does not allow (a closed product), ask for a separate licence.

Laya (Apache-2.0) and the LLM you use keep their own licences. Contributions: see [`CONTRIBUTING.md`](CONTRIBUTING.md).

If this work helps yours, please cite it:
*Tienef, "flickering-sign: an autonomous brain in three levels", 2026.
doi:[10.5281/zenodo.23041524](https://doi.org/10.5281/zenodo.23041524)* (all versions).
