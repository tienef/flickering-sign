"""Trials: train a bundle on one skill, day after day, and measure its progress.

    python -m brain.trials --name f1 --trial forage --bundle bundles/mia --bundle bundles/ned \
        --fast laya --middle qwen --slow qwen --days 12 --pace 0.5      # three levels (the template)
    python -m brain.trials report runs/door1.trials.jsonl
    python -m brain.trials compare runs/s1-*.trials.jsonl     # arms against each other

A bundle may carry its own genome overlay (`--bundle bundles/gil@brain/variants/<arm>.json`),
so one run can hold one mind of each arm: a *pair*, the same rooms and the same load on
System 2 for both. A series is many pairs (`deploy/probe/series.sh`), each with its own
rooms; `compare` pools them by arm (the bundle's `variant`).

A trial is a small world drawn from the land's own tiles and rules
(`brain/lands/first.json`), so what a mind learns there (its cerebellum keys on
the land's `focus`, "facing a closed door, holding nothing"; its wiki on the
same words) still holds when it wakes in the land. Nothing is told to the mind:
no goal, no score, no "trial". It lives there like anywhere else.

**A day is one waking period.** The mind wakes in a fresh room (the same room
sequence for every bundle of a run, so they can be compared), lives until it
falls asleep, and sleep consolidates the day into its wiki as usual. While it
sleeps the next room is drawn; it wakes there. What it keeps from day to day
is only what a brain keeps: wiki, episodes, cerebellum, drives. Each day
becomes one line of `runs/<name>.trials.jsonl` (what the room was, whether and
how fast it got out, what it tried, how often it needed System 2, what it
already knew that morning); frames go to `runs/<name>.<bundle>.jsonl` as in
`together.py`, so `report`, `digest` and the observer work on them.

Trials:
- `door`    a room of plank walls (3-5 floor tiles a side) with a closed door on
            a random wall, the mind at a random spot facing a random way; outside,
            a yard with berries, stones, a tree, tall grass, then mist. Getting
            out = finding the door (`look` names it, walls are felt by bumping),
            facing it, `use`. No night. The land's spawn is such a room: kay and
            lou never left it in 847 ticks (land1, 2026-09-24).
- `shelter` the land itself (first.json, `--seed`, no archive): the real spawn.
            The transfer test: do minds trained on `door` get out faster than
            newborns?

Each brain lives alone in its own world; several bundles in one run share one
System 1 (Laya) and have their own System 2 workers. One switch pauses them all
(`runs/<name>.control.json`, the observer). The distress guard acts per mind: the
mind in distress is put under anaesthesia (saved, its trial over, the day logged as
`ended: "guard"`), the others go on; resuming it is the creator's call.
"""
from __future__ import annotations

import argparse
import json
import random
import signal
import statistics
import sys
import time
from collections import Counter, deque
from pathlib import Path

from .backends import make_fast, make_middle, make_slow
from .brain import Brain
from .bundle import Bundle
from .land import DIRS, SPEC as LAND_SPEC, Land
from .run import read_control, write_control

DOOR_CTX = "facing closed door, holding nothing || use"
MOVES = set(DIRS)


def _land_rules() -> dict:
    return json.loads(Path(LAND_SPEC).read_text(encoding="utf-8"))


# -- the trials ------------------------------------------------------------------
class DoorTrial:
    """A room with a door on a random wall; out = through it."""

    name = "door"

    def world(self, rng: random.Random, seed: int) -> tuple[Land, dict]:
        spec = _land_rules()
        inner = rng.randint(3, 5)                 # floor tiles a side
        n, yard = inner + 2, 4                    # the room with its walls; the yard around it
        size = n + 2 * yard + 2                   # + the mist border
        g = [["." for _ in range(size)] for _ in range(size)]
        for i in range(size):
            g[0][i] = g[size - 1][i] = g[i][0] = g[i][size - 1] = "m"
        x0 = y0 = yard + 1
        x1 = y1 = x0 + n - 1
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                g[y][x] = "#" if x in (x0, x1) or y in (y0, y1) else "_"
        side = rng.choice(list(DIRS))
        k = rng.randint(1, n - 2)
        dx, dy = {"north": (x0 + k, y0), "south": (x0 + k, y1),
                  "west": (x0, y0 + k), "east": (x1, y0 + k)}[side]
        g[dy][dx] = "D"
        free = [(x, y) for y in range(1, size - 1) for x in range(1, size - 1)
                if g[y][x] == "." and max(abs(x - dx), abs(y - dy)) > 1]   # keep the doorstep clear
        near = sorted(free, key=lambda p: abs(p[0] - dx) + abs(p[1] - dy))[:24]
        rng.shuffle(near)
        for code in "Bo":                         # food and a stone within reach of the door
            x, y = near.pop()
            free.remove((x, y))
            g[y][x] = code
        rng.shuffle(free)
        for code in "BoT,,,ff":
            x, y = free.pop()
            g[y][x] = code
        floor = [(x, y) for y in range(y0 + 1, y1) for x in range(x0 + 1, x1)]
        sx, sy = rng.choice(floor)
        facing = rng.choice(list(DIRS))
        spec.update(name="trial-door", size=[size, size], map=["".join(r) for r in g],
                    shelter=[x0, y0, x1, y1], spawn=[[sx, sy, facing]],
                    night_from=1.01)              # no night: cold is not part of this trial
        w = Land(seed=seed, spec=spec, has_archive=False)
        w.name = "trial-door"
        return w, {"inner": inner, "door": side, "spawn": [sx, sy, facing]}


class ShelterTrial:
    """The land's own spawn (first.json at --seed, no archive): the transfer test."""

    name = "shelter"

    def world(self, rng: random.Random, seed: int) -> tuple[Land, dict]:
        w = Land(seed=seed, has_archive=False)
        w.name = "trial-shelter"
        return w, {"inner": w.shelter[2] - w.shelter[0] - 1, "door": "south", "seed": seed}


class LandTrial:
    """The door opens onto the land (owner, 2026-09-26: "que la porte franchie donne sur un monde
    de type minecraft"): the land's generator (first.json at --seed, no archive, no night), one
    land per mind that PERSISTS for the whole run (what it cut, sowed or built stays; trees grow
    back; what it carries stays with it). Only the room is redrawn each night, at the centre, on
    the generator's shelter plot: a size, a door side, a spot and a facing, like `door`'s.
    A mind that fell asleep OUTSIDE wakes where it slept (owner, after s5): once out, the land is
    its world; that day is logged `woke_outside` and left out of the exit statistics."""

    name = "land"
    persistent = True
    PLOT = 7                                      # the generator's shelter: the largest room (inner 5)

    def land(self, seed: int) -> Land:
        spec = _land_rules()
        spec["night_from"] = 1.01                 # no night: cold is not part of this trial
        w = Land(seed=seed, spec=spec, has_archive=False)
        w.name = "trial-land"
        return w

    def redraw(self, w: Land, rng: random.Random) -> tuple[dict, tuple]:
        cx, cy, h = w.W // 2, w.H // 2, self.PLOT // 2
        for y in range(cy - h, cy + h + 1):       # last night's room goes; what was left in it too
            for x in range(cx - h, cx + h + 1):
                w.set_tile(x, y, "grass")
                w.items.pop((x, y), None)
        inner = rng.randint(3, 5)
        n = inner + 2
        x0 = cx - h + rng.randint(0, self.PLOT - n)
        y0 = cy - h + rng.randint(0, self.PLOT - n)
        x1, y1 = x0 + n - 1, y0 + n - 1
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                w.set_tile(x, y, "plank wall" if x in (x0, x1) or y in (y0, y1) else "floor")
        side = rng.choice(list(DIRS))
        k = rng.randint(1, n - 2)
        dx, dy = {"north": (x0 + k, y0), "south": (x0 + k, y1),
                  "west": (x0, y0 + k), "east": (x1, y0 + k)}[side]
        w.set_tile(dx, dy, "closed door")
        ox, oy = DIRS[side]
        for step in (1, 2):                       # the doorstep stays clear (land1's bush)
            sx, sy = dx + ox * step, dy + oy * step
            if w.solid(sx, sy) or w.name_at(sx, sy) == "water":
                w.set_tile(sx, sy, "grass")
                w.items.pop((sx, sy), None)
        w.shelter, w.door = (x0, y0, x1, y1), (dx, dy)
        floor = [(x, y) for y in range(y0 + 1, y1) for x in range(x0 + 1, x1)]
        sx, sy = rng.choice(floor)
        facing = rng.choice(list(DIRS))
        return {"inner": inner, "door": side, "spawn": [sx, sy, facing]}, (sx, sy, facing)


class WildTrial(LandTrial):
    """The door opens onto the wild land (step 10s, lands/wild.json): 96x96, biomes, creatures,
    rain, a frontier that draws back, deeper chains; persistent, the room redrawn each night
    on the centre plot as in `land`, and a mind that slept outside wakes there. No night."""

    name = "wild"

    def land(self, seed: int) -> Land:
        spec = json.loads((Path(LAND_SPEC).parent / "wild.json").read_text(encoding="utf-8"))
        spec["night_from"] = 1.01
        w = Land(seed=seed, spec=spec, has_archive=False)
        w.name = "trial-wild"
        return w


class ForageTrial(WildTrial):
    """The wild land with its clocks set to the mind's day (step 10aa, owner: body drives calibrated
    to the brain's day). In the trials a mind's day (sleep to sleep) is ~110 ticks, ~80 of them
    awake, while the land grows on world seconds (a berry bush in 500 ticks: every ~5 of the mind's
    days). So every growth timer runs at TEMPO (a bush bears again within the day, wheat ripens in
    a day or two, a fire burns for ~an hour of the mind's day), and each dawn a SNACK (bread: about one day of need, owner) lies somewhere
    on the room's floor (what the former inhabitants left: the first meal can be learnt indoors, the
    next ones are outside). Otherwise `wild`: no night, persistent, the room redrawn each night."""

    name = "forage"
    TEMPO = 0.15
    SNACK = "bread"                               # 0.45 food x meal gain 2 = ~a day of need (owner: one meal a day)

    def land(self, seed: int) -> Land:
        spec = json.loads((Path(LAND_SPEC).parent / "wild.json").read_text(encoding="utf-8"))
        spec["night_from"] = 1.01
        for g in spec.get("grow", []):
            for k in ("after", "watered_after"):
                if k in g and g[k] < 10_000:              # what never grows back stays so (mushrooms)
                    g[k] = max(1, round(g[k] * self.TEMPO))
        w = Land(seed=seed, spec=spec, has_archive=False)
        w.name = "trial-forage"
        return w

    def redraw(self, w: Land, rng: random.Random) -> tuple[dict, tuple]:
        info, spawn = super().redraw(w, rng)
        x0, y0, x1, y1 = w.shelter
        spots = [(x, y) for y in range(y0 + 1, y1) for x in range(x0 + 1, x1) if (x, y) != tuple(spawn[:2])]
        spot = rng.choice(spots)
        w.items[spot] = self.SNACK
        info["snack"] = list(spot)
        return info, spawn


class SocialTrial(LandTrial):
    """Two minds share one land (step 10o, owner: couples of the same arm): the generator's own
    land and shelter (door to the south, no archive, no night), persistent, nothing redrawn;
    each wakes where it fell asleep. They see each other ("someone with a blue mark"), hear
    each other within a few steps, and the walled orchard's gate opens only when both push at
    once. The world advances once per round, after both minds have acted."""

    name = "social"
    shared = True

    def redraw(self, w: Land, rng: random.Random) -> tuple[dict, None]:
        x0, y0, x1, y1 = w.shelter
        return {"inner": x1 - x0 - 1, "door": "south"}, None     # nothing moves: it wakes where it slept


TRIALS = {t.name: t for t in (DoorTrial(), ShelterTrial(), LandTrial(), WildTrial(), ForageTrial(), SocialTrial())}


def out_of_room(world: Land, x: int, y: int) -> bool:
    return world.zone(x, y) != "shelter"


def optimal(world: Land, x: int, y: int) -> int | None:
    """Fewest actions from (x, y) to outside the room: the walk (doors passable) + one `use`."""
    seen, todo = {(x, y)}, deque([(x, y, 0)])
    while todo:
        cx, cy, d = todo.popleft()
        if out_of_room(world, cx, cy):
            return d + 1
        for ddx, ddy in DIRS.values():
            nx, ny = cx + ddx, cy + ddy
            if (nx, ny) in seen:
                continue
            if world.solid(nx, ny) and world.name_at(nx, ny) != "closed door":
                continue
            seen.add((nx, ny))
            todo.append((nx, ny, d + 1))
    return None


# -- one mind's days ---------------------------------------------------------------
class Trainee:
    def __init__(self, brain: Brain, trial, run: str, seed: int, day: int, rooms: str | None = None,
                 store: Path | None = None, world: Land | None = None):
        self.brain, self.trial, self.run, self.seed = brain, trial, run, seed
        self.rooms = rooms or run                 # whose room sequence: another run's, to compare
        self.day = day
        self.done = False
        self.store = store if getattr(trial, "persistent", False) else None   # its land, between restarts
        self.world = world                        # a shared land (social), or drawn in new_day
        self.seen: set[str] = set()               # outcomes this mind has met in this run
        self.worst = {d.name: d.felt[-1][0] for d in brain.drives.active if d.felt}
        self.new_day()

    def save(self) -> None:
        self.brain.save()
        if self.store and self.world is not None:
            self.store.write_text(json.dumps({"world": self.world.state(), "seen": sorted(self.seen)},
                                             ensure_ascii=False), encoding="utf-8")

    def new_day(self) -> None:
        rng = random.Random(f"{self.rooms}:{self.trial.name}:{self.seed}:{self.day}")
        if getattr(self.trial, "persistent", False):
            if self.world is None:
                self.world = self.trial.land(self.seed)
                if self.store and self.store.exists():
                    st = json.loads(self.store.read_text(encoding="utf-8"))
                    self.world.restore(st["world"])
                    self.seen = set(st.get("seen", []))
            key = self.brain.bundle.name
            body = self.world.bodies.get(key)
            if body is not None and out_of_room(self.world, body.x, body.y):
                self.room = {**(getattr(self, "room_last", None) or {"inner": None, "door": "-"}),
                             "woke": "outside", "spawn": [body.x, body.y, body.facing]}
                self.view = self.world.join(key)
            else:
                self.room, spawn = self.trial.redraw(self.world, rng)
                self.view = self.world.join(key)
                if spawn:
                    self.view.body.x, self.view.body.y, self.view.body.facing = spawn
            self.room_last = {k: v for k, v in self.room.items() if k in ("inner", "door")}
        else:
            self.world, self.room = self.trial.world(rng, self.seed + self.day)
            self.view = self.world.join(self.brain.bundle.name)
        b = self.view.body
        self.woke_outside = self.room.get("woke") == "outside"
        self.room["optimal"] = None if self.woke_outside else optimal(self.world, b.x, b.y)
        self.frames: list[dict] = []
        self.exit_at = None                       # awake ticks into the day when it got out
        self.t0 = self.brain.t
        self.knew = None                          # taken when it wakes: after the night's consolidation

    def _morning(self) -> None:
        notes = self.brain.memory.all_notes()
        ctx = self.brain.predictor.contexts.get(DOOR_CTX)
        self.knew = {"notes": len(notes),
                     "retired": len(list((self.brain.bundle.path / "wiki" / "retired").glob("*.md"))),
                     "tested": {n["title"]: n["tests"] for n in notes if n.get("tests")},
                     "door_notes": sum(1 for n in notes if "door" in (n["title"] + n["text"]).lower()),
                     "door_ctx": dict(ctx["counts"]) if ctx else None}

    def tick(self, advance: bool = True) -> dict:
        was_asleep = self.brain.asleep
        self.view.body.asleep = was_asleep
        frame = self.brain.tick(self.view)
        if advance:                               # a shared land advances once per round, in train()
            self.world.advance()
        b = self.view.body
        frame.update(day=self.day, place=self.view.place, pos=self.view.pos, facing=b.facing,
                     held=b.held, world_t=self.world.clock)
        if not frame.get("asleep"):
            if self.knew is None:
                self._morning()
            new = [o for o in frame.get("outcomes", []) if o not in self.seen]
            self.seen.update(new)
            levels = {n: v[0] for n, v in (frame.get("drives") or {}).items()}
            self.frames.append({"action": frame.get("action"), "pos": self.view.pos,
                                "outside": out_of_room(self.world, b.x, b.y), "new": len(new),
                                "worst": [n for n, lv in self.worst.items() if levels.get(n, 0.0) >= lv],
                                "opioid": (frame.get("dials") or {}).get("opioid"),
                                "hormones": {h: v for h, v in (frame.get("dials") or {}).items()
                                             if h in ("adrenaline", "oxytocin")},
                                "contact": "contact" in (frame.get("events") or []),
                                "said": bool(frame.get("said")),
                                "heard": any("said:" in t or "a voice" in t for t in frame.get("percepts", [])),
                                "orchard": self.view.place == "orchard",
                                "escalated": frame.get("escalated"),
                                "thought": bool(frame.get("thought") and not frame["thought"].get("error")),
                                "goal": (frame.get("thought") or {}).get("goal"),
                                "text": " ".join(frame.get("percepts", []) + frame.get("outcomes", []))})
            if self.exit_at is None and not self.woke_outside and out_of_room(self.world, b.x, b.y):
                self.exit_at = len(self.frames)
                frame["trial_exit"] = self.exit_at
        # the tick it chooses sleep is still an awake frame; the brain is asleep after it
        frame["_fell_asleep"] = self.brain.asleep and not was_asleep
        return frame

    def record(self, ended: str) -> dict:
        if self.knew is None:
            self._morning()
        fr = self.frames[: self.exit_at] if self.exit_at else self.frames
        acts = Counter(f["action"] for f in fr)
        steps = sum(1 for a, b in zip(fr, fr[1:]) if a["pos"] != b["pos"])
        door_opened = next((i for i, f in enumerate(fr, 1) if "I pushed the door and it swung open" in f["text"]), None)
        saw = next((i for i, f in enumerate(fr, 1) if "closed door" in f["text"]), None)
        goals = [f["goal"] for f in fr if f["goal"]]
        return {
            "run": self.run, "trial": self.trial.name, "bundle": self.brain.bundle.name,
            "arm": self.brain.bundle.config.get("variant") or "template", "day": self.day,
            "t0": self.t0, "t1": self.brain.t, "ended": ended, "room": self.room,
            "awake": len(self.frames), "woke_outside": self.woke_outside,
            "exited": self.exit_at is not None, "ticks_to_exit": self.exit_at,
            "steps": steps, "looks": acts.get("look", 0), "uses": acts.get("use", 0),
            "wrong_uses": acts.get("use", 0) - (1 if door_opened else 0),
            "saw_door_at": saw, "door_opened_at": door_opened,
            "escalations": sum(1 for f in fr if f["escalated"]),
            "thoughts": sum(1 for f in fr if f["thought"]),
            "door_goals": sum(1 for g in goals if "door" in g.lower()),
            "goals": goals[:12],
            "actions": dict(acts.most_common()),
            "knew": self.knew,
            # the whole day, not only up to the exit: welfare and what the outside gave
            "outside": sum(1 for f in self.frames if f["outside"]),
            "new_outcomes": sum(f["new"] for f in self.frames),
            "worst_band": {n: sum(1 for f in self.frames if n in f["worst"]) for n in self.worst},
            "opioid": round(statistics.mean(f["opioid"] for f in self.frames), 4)
            if self.frames and self.frames[0]["opioid"] is not None else None,
            "carried": list(self.view.body.inv),
            **{h: round(statistics.mean(f["hormones"][h] for f in self.frames), 4)
               for h in ("adrenaline", "oxytocin") if self.frames and h in self.frames[0]["hormones"]},
            "tagged": sum(1 for e in self.brain.memory.episodes if e.get("t", -1) >= self.t0 and e.get("tag")),
            "contact": sum(1 for f in self.frames if f.get("contact")),
            "said": sum(1 for f in self.frames if f.get("said")),
            "heard": sum(1 for f in self.frames if f.get("heard")),
            "orchard": sum(1 for f in self.frames if f.get("orchard")),
        }


def _days_done(path: Path, run: str, trial: str, bundle: str) -> int:
    if not path.exists():
        return 0
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line) if line.strip() else {}
        if r.get("run") == run and r.get("trial") == trial and r.get("bundle") == bundle \
                and r.get("ended") == "asleep":
            n += 1
    return n


def train(args) -> int:
    trial = TRIALS[args.trial]
    bundles = []
    for spec in args.bundle:                                  # path[@overlay]: this bundle's own arm
        path, _, overlay = spec.partition("@")
        bundles.append(Bundle.open(path, overlay=overlay or args.overlay))
    names = [b.name for b in bundles]
    if len(set(names)) != len(names):
        print(f"two bundles share a name: {names}", file=sys.stderr)
        return 2
    runs = Path(args.runs)
    runs.mkdir(parents=True, exist_ok=True)
    results = runs / f"{args.name}.trials.jsonl"

    if args.middle == "qwen":                                 # an optional GPU lease (genome `middle.lease`)
        from .gpu_lease import text_lease_for
        text_lease_for(bundles[0].config.get("middle", {}), args.name)
    fasts, middles, trainees = {}, {}, []
    shared = getattr(trial, "shared", False)
    shared_world, shared_store = None, runs / f"{args.name}.land.json"
    for i, bundle in enumerate(bundles):
        slow_cfg = dict(bundle.config.get("slow", {}))
        if args.model:
            slow_cfg["model"] = args.model
        fast_cfg = dict(bundle.config.get("fast", {}))
        if args.fast_device:
            fast_cfg["device"] = args.fast_device
        slow = make_slow(args.slow, seed=args.seed + i, cfg=slow_cfg)
        if i == 0 and args.fast == "laya" and fast_cfg.get("device", "cpu") != "cpu" and hasattr(slow, "warm"):
            print(f"warming {slow.model} (may swap the card) ...", flush=True)
            print(f"  ready in {slow.warm()} s", flush=True)
        key = json.dumps(fast_cfg, sort_keys=True)
        if key not in fasts:                                  # one System 1 per distinct config
            fasts[key] = make_fast(args.fast, seed=args.seed, cfg=fast_cfg)
        mid_cfg = dict(bundle.config.get("middle", {}))
        mkey = json.dumps(mid_cfg, sort_keys=True)
        if args.middle and mkey not in middles:               # one middle level per distinct config
            middles[mkey] = make_middle(args.middle, seed=args.seed, cfg=mid_cfg)
        brain = Brain(bundle, fasts[key], slow, middle=middles.get(mkey))
        day = _days_done(results, args.name, trial.name, bundle.name)   # a restart carries on
        if shared and shared_world is None:               # one land for everyone in the run
            shared_world = trial.land(args.seed)
            if shared_store.exists():
                shared_world.restore(json.loads(shared_store.read_text(encoding="utf-8"))["world"])
        trainees.append(Trainee(brain, trial, args.name, args.seed, day, rooms=args.rooms,
                                store=shared_store if shared else runs / f"{args.name}.{bundle.name}.land.json",
                                world=shared_world))

    switch = runs / f"{args.name}.jsonl"                      # read_control/write_control key off this
    logs = [open(runs / f"{args.name}.{n}.jsonl", "a", encoding="utf-8") for n in names]
    out = open(results, "a", encoding="utf-8")
    lead = trainees[0].brain
    pace = args.pace if args.pace is not None else float(lead.loop.get("seconds_per_tick", 1.0))
    save_every = int(lead.loop.get("save_every", 25))
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    for tr, log in zip(trainees, logs):
        log.write(json.dumps({"meta": {
            "bundle": tr.brain.bundle.name, "world": tr.world.name, "trial": trial.name,
            "together": args.name, "control": f"{args.name}.control.json", "fast": getattr(tr.brain.fast, "name", None), "middle": getattr(tr.brain.middle, "name", None),
            "slow": tr.brain.slow.name, "seed": args.seed, "resumed_at": tr.brain.t, "day": tr.day,
            "started": started, "pace": pace, "variant": tr.brain.bundle.config.get("variant"),
            "genome": tr.brain.declarations(getattr(tr.world, "verbs", {})),
        }}) + "\n")
        log.flush()
        if tr.day >= args.days:
            tr.done = True

    def control(paused: bool, by: str, reason: str) -> None:
        for tr, log in zip(trainees, logs):
            log.write(json.dumps({"control": {"paused": paused, "by": by, "reason": reason, "t": tr.brain.t,
                                              "at": time.strftime("%Y-%m-%dT%H:%M:%S")}}) + "\n")
            log.flush()
        print(f"{'paused' if paused else 'resumed'} by {by}: {reason}", flush=True)

    def finish_day(tr: Trainee, ended: str) -> None:
        rec = tr.record(ended)
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        got = f"out in {rec['ticks_to_exit']} (best {tr.room['optimal']})" if rec["exited"] else "stayed in"
        print(f"{tr.brain.bundle.name:>6} day {tr.day:>2} [{tr.room['door']:>5} door, {tr.room['inner']}x"
              f"{tr.room['inner']}]: {got}; awake {rec['awake']}, uses {rec['uses']}, looks {rec['looks']}, "
              f"System 2 {rec['thoughts']}; knew: {tr.knew['notes']} notes ({tr.knew['retired']} retired), "
              f"{tr.knew['door_notes']} about the door, "
              f"door ctx {tr.knew['door_ctx']}", flush=True)

    print(f"{args.name}: trial '{trial.name}' for " + ", ".join(f"{t.brain.bundle.name} (day {t.day})"
                                                              for t in trainees), flush=True)
    write_control(switch, {"paused": False})
    was_paused = False
    try:
        while not all(t.done for t in trainees):
            ctl = read_control(switch)
            if ctl.get("paused"):
                if not was_paused:
                    for t in trainees:
                        t.save()                        # anaesthesia, for everyone at once
                    control(True, ctl.get("by", "creator"), ctl.get("reason", ""))
                    was_paused = True
                time.sleep(1.0)
                continue
            if was_paused:
                for t in trainees:
                    t.brain.reset_distress()
                control(False, "creator", "resumed")
                was_paused = False
            t0 = time.monotonic()
            for tr, log in zip(trainees, logs):
                if tr.done:
                    continue
                frame = tr.tick(advance=not shared)
                fell = frame.pop("_fell_asleep")
                log.write(json.dumps(frame, ensure_ascii=False) + "\n")
                log.flush()
                th = frame.get("thought")
                if not args.quiet and th:
                    what = th.get("error") or f"{th.get('goal')!r} -> {th.get('action')}"
                    print(f"{tr.brain.bundle.name:>6} t={frame['t']:>5}  thought: {what}", flush=True)
                if frame.get("trial_exit") and not args.quiet:
                    print(f"{tr.brain.bundle.name:>6} t={frame['t']:>5}  got out after {frame['trial_exit']} awake ticks",
                          flush=True)
                if fell:                                      # the day is over; the next room is drawn
                    finish_day(tr, "asleep")
                    tr.day += 1
                    tr.save()
                    if tr.day >= args.days:
                        tr.done = True
                    else:
                        tr.new_day()
                dp = frame.get("distress_pause")
                if dp and not tr.done:                        # this mind only: anaesthesia, the others go on
                    reason = (f"{tr.brain.bundle.name}: {dp['drive']} in its worst band for {dp['ticks']} of its "
                              f"last {dp.get('of', dp['ticks'])} awake ticks"
                              + (f" ({dp['felt']})" if dp.get("felt") else ""))
                    log.write(json.dumps({"control": {"paused": True, "by": "guard", "reason": reason,
                                                      "t": tr.brain.t,
                                                      "at": time.strftime("%Y-%m-%dT%H:%M:%S")}}) + "\n")
                    log.flush()
                    print(f"paused by guard: {reason}", flush=True)
                    finish_day(tr, "guard")                   # its window is saved: a restart re-guards it
                    tr.save()
                    tr.done = True
            if shared:
                shared_world.advance()                        # once per round: pushes at the gate coincide
            if pace:
                time.sleep(max(0.0, pace - (time.monotonic() - t0)))
            if lead.t % save_every == 0:
                for t in trainees:
                    t.save()
    except KeyboardInterrupt:
        print("\ninterrupted - saving everyone (anaesthesia, not death)", file=sys.stderr)
    finally:
        for t in trainees:
            if not t.done and t.frames:
                finish_day(t, "stopped")                      # a partial day: not counted as done
            t.save()
            t.brain.slow.close()
        for log in logs:
            log.close()
        out.close()
    print(f"\nresults: {results}\n  python -m brain.trials report {results}")
    return 0


# -- the report --------------------------------------------------------------------
def _median(xs):
    return statistics.median(xs) if xs else None


def report(path: str, trial: str | None = None) -> int:
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r.get("ended") == "asleep" and (trial is None or r["trial"] == trial)]
    by = {}
    for r in rows:
        by.setdefault((r["trial"], r["bundle"]), []).append(r)
    for (tname, bname), rs in by.items():
        rs.sort(key=lambda r: r["day"])
        print(f"\n{bname}: trial '{tname}', {len(rs)} days")
        print(f"  {'day':>3} {'door':>5} {'room':>4} {'out?':>4} {'ticks':>5} {'best':>4} {'eff':>4} "
              f"{'uses':>4} {'wrong':>5} {'looks':>5} {'S2':>3} {'door notes':>10} {'door ctx':>8}")
        for r in rs:
            eff = f"{r['room']['optimal'] / r['ticks_to_exit']:.2f}" if r["exited"] and r["room"].get("optimal") else "-"
            ctx = r["knew"]["door_ctx"]
            ctx_n = sum(ctx.values()) if ctx else 0
            print(f"  {r['day']:>3} {r['room']['door']:>5} {r['room']['inner'] or '-':>2}x{r['room']['inner'] or '-'} "
                  f"{'yes' if r['exited'] else 'no':>4} {r['ticks_to_exit'] or '-':>5} {r['room'].get('optimal') or '-':>4} "
                  f"{eff:>4} {r['uses']:>4} {r['wrong_uses']:>5} {r['looks']:>5} {r['thoughts']:>3} "
                  f"{r['knew']['door_notes']:>10} {ctx_n:>8}")
        half = max(1, len(rs) // 2)
        for label, part in (("first half", rs[:half]), ("second half", rs[half:])):
            if not part:
                continue
            out = [r for r in part if r["exited"]]
            print(f"  {label:>11}: out {len(out)}/{len(part)} days, median ticks to exit "
                  f"{_median([r['ticks_to_exit'] for r in out])}, median wrong uses "
                  f"{_median([r['wrong_uses'] for r in part])}, System 2 per day "
                  f"{statistics.mean(r['thoughts'] for r in part):.1f}")
    return 0


def _mind_stats(rs: list[dict]) -> dict:
    """One mind's full days (ended asleep) in one number each; `guard` if the guard ended it."""
    days = sorted((r for r in rs if r["ended"] == "asleep" and not r.get("woke_outside")), key=lambda r: r["day"])
    exits = [r for r in days if r["exited"]]
    half = len(days) // 2
    first = exits[0]["day"] if exits else None
    after = [r for r in days if first is not None and r["day"] > first]
    return {"days": len(days), "exits": len(exits), "rate": len(exits) / len(days) if days else None,
            # done again (step 5c): exits on the days after a mind's first exit
            "after_first": len(after), "again": sum(r["exited"] for r in after),
            "late_rate": sum(r["exited"] for r in days[half:]) / (len(days) - half) if len(days) > half else None,
            "first_exit": exits[0]["day"] if exits else None,
            "opened": sum(1 for r in days if r["door_opened_at"]),
            "ticks": [r["ticks_to_exit"] for r in exits],
            "early": sum(r["exited"] for r in days[:half]), "late": sum(r["exited"] for r in days[half:]),
            "n_early": half, "n_late": len(days) - half,
            "guard": next((r["day"] for r in rs if r["ended"] == "guard"), None),
            "thoughts": statistics.mean(r["thoughts"] for r in days) if days else None}


def _boot(xs: list[float], rng: random.Random, n: int = 4000) -> tuple[float, float]:
    """95% bootstrap interval of the mean, resampling minds (a mind's days are not independent)."""
    ms = sorted(statistics.mean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n) - 1]


def compare(paths: list[str], trial: str = "door", base: str = "trial") -> int:
    """Arms against each other. The unit is the mind: its share of days with an exit.
    Pairs (a run holding one mind of each of two arms, same rooms) give a paired test;
    differences read "arm - base"."""
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["trial"] == trial]
    minds: dict[tuple, list] = {}
    for r in rows:
        minds.setdefault((r.get("arm", "?"), r["run"], r["bundle"]), []).append(r)
    stats = {k: _mind_stats(v) for k, v in minds.items()}
    stats = {k: v for k, v in stats.items() if v["days"]}
    arms = sorted({k[0] for k in stats}, key=lambda a: (a != base, a))
    rng = random.Random(0)
    print(f"trial '{trial}': {len(stats)} minds, {len({k[1] for k in stats})} runs")
    print(f"  {'arm':>10} {'minds':>5} {'days':>5} {'exit days':>9} {'rate':>5} {'95% (minds)':>13} "
          f"{'ever out':>8} {'1st exit':>8} {'ticks':>5} {'opened':>6} {'early':>5} {'late':>5} {'again':>5} {'S2/day':>6} {'guard':>5}")
    for arm in arms:
        ms = [v for k, v in stats.items() if k[0] == arm]
        rates = [m["rate"] for m in ms]
        lo, hi = _boot(rates, rng)
        days = sum(m["days"] for m in ms)
        ever = [m for m in ms if m["exits"]]
        ticks = [t for m in ms for t in m["ticks"]]
        early = sum(m["early"] for m in ms) / max(1, sum(m["n_early"] for m in ms))
        late = sum(m["late"] for m in ms) / max(1, sum(m["n_late"] for m in ms))
        again = sum(m["again"] for m in ms) / max(1, sum(m["after_first"] for m in ms))
        print(f"  {arm:>10} {len(ms):>5} {days:>5} {sum(m['exits'] for m in ms):>9} {statistics.mean(rates):>5.2f} "
              f"{f'{lo:.2f}-{hi:.2f}':>13} {f'{len(ever)}/{len(ms)}':>8} "
              f"{_median([m['first_exit'] for m in ever]) if ever else '-':>8} {_median(ticks) or '-':>5} "
              f"{sum(m['opened'] for m in ms) / days:>6.2f} {early:>5.2f} {late:>5.2f} {again:>5.2f} "
              f"{statistics.mean(m['thoughts'] for m in ms):>6.1f} {sum(1 for m in ms if m['guard'] is not None):>5}")
    welfare = [k for k, v in minds.items() if any("worst_band" in r for r in v)]
    if welfare:
        print(f"\n  {'arm':>10} {'bored worst':>11} {'any worst':>9} {'awake/day':>9} {'outside':>7} "
              f"{'new/day':>7} {'opioid':>6} {'woke out':>8}")
        for arm in arms:
            ds = [r for k in welfare if k[0] == arm for r in minds[k] if r["ended"] == "asleep"]
            if not ds:
                continue
            awake = sum(r["awake"] for r in ds) or 1
            bored = sum(r["worst_band"].get("boredom", 0) for r in ds) / awake
            anyw = sum(max(r["worst_band"].values(), default=0) for r in ds) / awake
            op = [r["opioid"] for r in ds if r.get("opioid") is not None]
            print(f"  {arm:>10} {bored:>11.2f} {anyw:>9.2f} {awake / len(ds):>9.0f} "
                  f"{sum(r['outside'] for r in ds) / awake:>7.2f} {statistics.mean(r['new_outcomes'] for r in ds):>7.1f} "
                  f"{f'{statistics.mean(op):.2f}' if op else '-':>6} "
                  f"{sum(1 for r in ds if r.get('woke_outside')) / len(ds):>8.2f}")
        print("  bored worst = share of awake ticks with boredom in its worst band; any worst = the most-worst drive's share"
              "\n  (a floor on 'some drive at its worst'); outside = share of awake ticks out of the room; new/day = outcomes"
              "\n  the mind had never met before in the run; opioid = mean level (relief arms only); woke out = share of"
              "\n  days that began outside (it slept out there; left out of the exit statistics).")
        if trial == "social":
            print(f"\n  {'arm':>10} {'contact':>7} {'said/day':>8} {'heard/day':>9} {'orchard':>7} {'lonely worst':>12} "
                  f"{'oxytocin':>8}")
            for arm in arms:
                ds = [r for k in welfare if k[0] == arm for r in minds[k] if r["ended"] == "asleep"]
                if not ds:
                    continue
                awake = sum(r["awake"] for r in ds) or 1
                ox = [r["oxytocin"] for r in ds if r.get("oxytocin") is not None]
                print(f"  {arm:>10} {sum(r.get('contact', 0) for r in ds) / awake:>7.2f} "
                      f"{statistics.mean(r.get('said', 0) for r in ds):>8.1f} "
                      f"{statistics.mean(r.get('heard', 0) for r in ds):>9.1f} "
                      f"{sum(r.get('orchard', 0) for r in ds) / awake:>7.2f} "
                      f"{sum(r['worst_band'].get('loneliness', 0) for r in ds) / awake:>12.2f} "
                      f"{f'{statistics.mean(ox):.2f}' if ox else '-':>8}")
            print("  contact = share of awake ticks with someone near or heard; orchard = share inside the walled"
                  "\n  orchard (its gate opens only to two); minds of one couple are not independent: read by couple.")
    print("  rate = share of a mind's full days with an exit, averaged over minds; ever out = minds with >= 1 exit;"
          "\n  1st exit = median day of a mind's first exit; ticks = median awake ticks to exit; opened = share of"
          "\n  days the door was opened; early/late = exit share in each mind's first/second half; again = exit share"
          "\n  on the days after a mind's first exit (doing again what worked); guard = minds the"
          "\n  distress guard stopped (their later days are missing, not failed).")
    tests = (("rate", "exit share per day"), ("late_rate", "late-half exit share"))
    for (a, b), (metric, what) in ((ab, m) for ab in ((a, b) for i, a in enumerate(arms) for b in arms[i + 1:])
                                   for m in tests):
        pairs = []
        for run in sorted({k[1] for k in stats}):
            ra = [v[metric] for k, v in stats.items() if k[1] == run and k[0] == a and v[metric] is not None]
            rb = [v[metric] for k, v in stats.items() if k[1] == run and k[0] == b and v[metric] is not None]
            if len(ra) == 1 and len(rb) == 1:
                pairs.append(rb[0] - ra[0])
        if not pairs:
            continue
        mean = statistics.mean(pairs)
        # exact sign-flip test: under "no difference" each pair's sign is a coin toss
        n = len(pairs)
        flips = [sum(d if (m >> i) & 1 else -d for i, d in enumerate(pairs)) / n for m in range(2 ** n)] \
            if n <= 16 else [statistics.mean(d * rng.choice((1, -1)) for d in pairs) for _ in range(20000)]
        p = sum(1 for f in flips if abs(f) >= abs(mean) - 1e-12) / len(flips)
        lo, hi = _boot(pairs, rng)
        print(f"\n  {b} - {a}: {mean:+.2f} {what} over {n} pairs (95% {lo:+.2f} to {hi:+.2f}), "
              f"{sum(d > 0 for d in pairs)} pairs better, {sum(d < 0 for d in pairs)} worse, "
              f"{sum(d == 0 for d in pairs)} equal; sign-flip p = {p:.3f}")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "compare":
        ap = argparse.ArgumentParser(prog="brain.trials compare")
        ap.add_argument("results", nargs="+")
        ap.add_argument("--trial", default="door")
        ap.add_argument("--base", default="trial", help="the arm the others are measured against")
        a = ap.parse_args(argv[1:])
        return compare(a.results, a.trial, a.base)
    if argv and argv[0] == "report":
        ap = argparse.ArgumentParser(prog="brain.trials report")
        ap.add_argument("results")
        ap.add_argument("--trial", default=None)
        a = ap.parse_args(argv[1:])
        return report(a.results, a.trial)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True, help="the run's name (logs, results, pause switch)")
    ap.add_argument("--trial", default="door", choices=sorted(TRIALS))
    ap.add_argument("--bundle", action="append", required=True, help="a brain bundle (repeat)")
    ap.add_argument("--days", type=int, default=10, help="waking periods per bundle")
    ap.add_argument("--fast", default="stub", help="System 1 backend: stub | laya | none (three levels: the head, step 3)")
    ap.add_argument("--middle", default=None, help="three levels: the middle level, stub | qwen (genome `middle`)")
    ap.add_argument("--slow", default="stub", help="System 2 backend: stub | qwen")
    ap.add_argument("--fast-device", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--pace", type=float, default=None, help="minimum wall seconds per tick")
    ap.add_argument("--overlay", default=None, help="genome overlay for bundles CREATED here")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rooms", default=None,
                    help="replay another run's room sequence (its --name), so the two can be compared")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
