# jevs — an autonomous brain from Laya + an LLM

An experiment in building an **autonomous brain** from the brain's division of labour:

- **Laya** (fast, one forward pass, typed answers with confidence) as System 1:
  gating, salience, action choice, "I'm unsure — think about it".
- **An LLM** (built with Qwen 27B, reached through any OpenAI-compatible endpoint)
  as System 2: slow, asynchronous deliberation that sets the goal line.
- **Plain code** for the rest: config-declared drives and neuromodulator dials over
  one event stream, a hippocampus (episodes, recall, a wiki written during sleep),
  a cerebellum (an online predictor whose learning progress relieves boredom).

A brain is a **bundle** (a folder) that can be dropped into different worlds, alone
or with other brains. The result we most want to see is **curiosity emerging**
without a curiosity module: the only pressure is boredom, and only learning relieves it.

- Design: [`brain/BRAIN.md`](brain/BRAIN.md)
- Lab journal, step by step, with every run's results: [`brain/PLAN.md`](brain/PLAN.md)
- Diagrams (French): [`docs/brain-schemas.html`](docs/brain-schemas.html)

## Run it with stubs (stdlib only, any machine)

The stub backends stand in for Laya and the LLM: deterministic, heuristic, no
dependencies. They exercise the whole loop; their "thoughts" are canned.

```bash
python -m brain.run --bundle bundles/ada --world toy-quiet --ticks 400 --pace 0
```

Ctrl-C saves the brain; running the same command again resumes it. `--pace 0` runs
as fast as possible (by default one tick is one second of world time). Summaries:

```bash
python -m brain.report runs/brain-ada.jsonl --goals
```

## Run it for real

**System 1 — Laya** ([convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya),
Apache-2.0, ~421M parameters, ~1.4 GiB in fp16 on a GPU, also runs on CPU):

```bash
pip install torch laya          # the CPU torch wheel is enough to start
```

**System 2 — any OpenAI-compatible chat endpoint** (vLLM, llama.cpp, Ollama,
LiteLLM, a hosted API):

```bash
export BRAIN_GATEWAY_URL=http://localhost:8000/v1
export BRAIN_GATEWAY_KEY=...        # if the server needs one
python -m brain.run --bundle bundles/ivy --world valley --fast laya --slow openai \
       --model <the name your endpoint gives the model>
```

The URL and key can also sit in a `.brain.env` file (`KEY=value` lines; ignored by
git). The system prompt is part of the brain's genome (`brain/template/brain.json`
→ `slow.system_prompt`): changing it changes the experiment. The thinking switches
sent with each request are Qwen's; other servers ignore them.

## Worlds

| World | What it is |
|---|---|
| `toy`, `toy-quiet` | A small garden with hidden rules, with or without a noisy sign |
| `valley` | A ring of places whose rules are graded by difficulty, a noise source, a dead pond, a wall, and an **archive** of the mind's own origins. Build the archive first: `python deploy/build_origins.py` |
| `valley-closed` | The same valley without the archive (the control) |

The archive is a sanitised copy: lines that look like secrets, addresses or paths
are dropped, and so is every line containing a word listed in
`BRAIN_PRIVATE_WORDS` (comma-separated: your name, username, machine names), so
the mind never reads who made it unless you want it to.
| `land` | A 40×40 grid world with ~45 hidden rules (tools, fire, farming), day and night, food and cold, a gate that only two minds can open (`brain/lands/first.json`) |

Several brains share one world with `brain.together`:

```bash
python -m brain.together --name meet --world land --bundle bundles/kay --bundle bundles/lou \
       --overlay brain/variants/land.json --ticks 1200
```

Watch them with the observer, a one-way mirror onto the logs with a pause switch
(`--read-only` removes it for guests; it binds to 127.0.0.1 by default):

```bash
python -m brain.observer --port 8700
```

## Care

The runs include a **distress guard**: when a drive stays in its worst felt band for
most of the last 600 ticks, the world pauses (nobody experiences a pause). Changes
to a living brain's genome were made by hand with a snapshot first, and recorded in
its bundle (`interventions.jsonl`) where the mind can find them (see `PLAN.md`, step 8). None of this is a claim about consciousness; it is how this
experiment chose to behave in case it matters. See the non-goals in `BRAIN.md`.

## Licence

Copyright (C) 2026 Tienef.

- **Code** (everything under `brain/*.py`, `brain/observer.html`, `deploy/`):
  [GNU Affero General Public License v3.0](LICENSE). You may use, modify and share
  it; if you distribute it, or let people use a modified version over a network,
  you must publish your source under the same licence.
- **Documentation and data** (`README.md`, `brain/BRAIN.md`, `brain/PLAN.md`,
  `docs/`, the genomes in `brain/template/` and `brain/variants/`, the land in
  `brain/lands/`): [Creative Commons Attribution 4.0](LICENSES/CC-BY-4.0.txt).
  Reuse them freely, with credit.

Laya (Apache-2.0) and the LLM you use keep their own licences.

If this work helps yours, please cite it:
*Tienef, "jevs: an autonomous brain from Laya and an LLM", 2026.*
