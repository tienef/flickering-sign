"""Several brains in one shared world, on one clock (step 8; lands, step 9).

    python -m brain.together --name meet1 --world valley \
        --bundle bundles/ivy --bundle bundles/jay --fast laya --slow qwen --ticks 3600

Each tick, every awake brain senses, decides and acts once, in a shuffled order;
what one does reaches the others at their next sense (no one hears the future,
and the order never decides who is heard). The world never names anyone: each
mind is "someone with a <colour> mark" to the others (worlds.Valley, shared).

One world, one clock, one switch: `runs/<name>.control.json` pauses every brain
and the world together, whoever writes it (the observer, the creator, or the
distress guard when ANY brain stays in its worst band too long). So no mind
ever experiences another freezing. Each brain keeps its own log
(`runs/<name>.<bundle>.jsonl`, the same rows as `run.py`, plus `with` and
`said`), so `report`, `digest` and the observer work per brain.

The world is kept in `worlds/<name>/world.json` (the objects, where each body
stands, its mark), so relations survive a power-off: start the same command
again and everyone carries on where they were.

Compute: System 1 is shared (one Laya per distinct fast config, used in turn);
System 2 is one worker per brain, and the engine's slots (the q27 brain
profile has 3) let their thoughts run side by side.
"""
from __future__ import annotations

import argparse
import json
import signal
import random
import sys
import time
from pathlib import Path

from .backends import make_fast, make_middle, make_slow
from .brain import Brain
from .bundle import Bundle
from .digest import write_digest
from .origins import load_archive
from .run import read_control, write_control
from .land import SPEC as LAND_SPEC, Land
from .worlds import Valley


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True, help="the shared world's name (logs, state, pause switch)")
    ap.add_argument("--world", default="valley", choices=["valley", "valley-closed", "land", "land-closed"])
    ap.add_argument("--land", default=None, help="a land's data file (default brain/lands/first.json)")
    ap.add_argument("--bundle", action="append", required=True, help="a brain bundle (repeat)")
    ap.add_argument("--ticks", type=int, default=300, help="ticks to run; 0 = until stopped")
    ap.add_argument("--fast", default="stub", help="System 1 backend: stub | laya | none (three levels: the head, step 3)")
    ap.add_argument("--middle", default=None, help="three levels: the middle level, stub | qwen (genome `middle`)")
    ap.add_argument("--slow", default="stub", help="System 2 backend: stub | qwen")
    ap.add_argument("--fast-device", default=None, help="override fast.device (cpu | cuda)")
    ap.add_argument("--think-budget", type=int, default=None)
    ap.add_argument("--model", default=None, help="override slow.model")
    ap.add_argument("--pace", type=float, default=None,
                    help="minimum wall seconds per world tick (default: loop.seconds_per_tick)")
    ap.add_argument("--overlay", default=None, help="genome overlay for bundles CREATED here")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--worlds", default="worlds")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    # `kill` (SIGTERM) saves like Ctrl-C: nohup'd background runs ignore SIGINT
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    bundles = [Bundle.open(b, overlay=args.overlay) for b in args.bundle]
    names = [b.name for b in bundles]
    if len(set(names)) != len(names):
        print(f"two bundles share a name: {names}", file=sys.stderr)
        return 2

    archive = args.world in ("valley", "land")
    if args.world.startswith("land"):
        world = Land(seed=args.seed, spec=args.land or LAND_SPEC, has_archive=archive)
    else:
        world = Valley(seed=args.seed, shared=True, has_archive=archive)
    wdir = Path(args.worlds) / args.name
    wdir.mkdir(parents=True, exist_ok=True)
    wstate = wdir / "world.json"
    if wstate.exists():
        world.restore(json.loads(wstate.read_text(encoding="utf-8")))

    def save_world() -> None:
        tmp = wstate.with_suffix(".tmp")
        tmp.write_text(json.dumps(world.state(), indent=1), encoding="utf-8")
        tmp.replace(wstate)

    if args.middle == "qwen":                                 # an optional GPU lease (genome `middle.lease`)
        from .gpu_lease import text_lease_for
        text_lease_for(bundles[0].config.get("middle", {}), args.name)
    fasts, middles, brains, views = {}, {}, [], []
    for i, bundle in enumerate(bundles):
        slow_cfg = dict(bundle.config.get("slow", {}))
        if args.model:
            slow_cfg["model"] = args.model
        if args.think_budget is not None:
            slow_cfg["think_budget"] = args.think_budget
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
        brains.append(Brain(bundle, fasts[key], slow, middle=middles.get(mkey)))
        papers = load_archive(bundle.path) if archive else None
        views.append(world.join(bundle.name, papers=papers))
    save_world()

    runs = Path(args.runs)
    runs.mkdir(parents=True, exist_ok=True)
    switch = runs / f"{args.name}.jsonl"                      # read_control/write_control key off this
    logs = [open(runs / f"{args.name}.{n}.jsonl", "a", encoding="utf-8") for n in names]
    # the world's own log (P3): the map once at each start, then one row per round (the inspector's map)
    wlog = open(runs / f"{args.name}.world.jsonl", "a", encoding="utf-8") if hasattr(world, "world_row") else None
    if wlog:
        wlog.write(json.dumps({**world.world_meta(), "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                               "marks": {v.body.key: v.body.mark for v in views}}, ensure_ascii=False) + "\n")
        wlog.flush()
    marks = {v.body.key: v.body.mark for v in views}
    lead = brains[0]
    pace = args.pace if args.pace is not None else float(lead.loop.get("seconds_per_tick", 1.0))
    save_every = int(lead.loop.get("save_every", 25))
    digest_every = int(lead.loop.get("digest_every", 86400))
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    for brain, view, log in zip(brains, views, logs):
        log.write(json.dumps({"meta": {
            "bundle": brain.bundle.name, "world": world.name, "together": args.name,
            "mark": marks[brain.bundle.name],
            "others": {n: marks[n] for n in names if n != brain.bundle.name},
            "control": f"{args.name}.control.json", "fast": getattr(brain.fast, "name", None), "middle": getattr(brain.middle, "name", None), "slow": brain.slow.name,
            "seed": args.seed, "resumed_at": brain.t, "started": started, "pace": pace,
            "variant": brain.bundle.config.get("variant"), "genome": brain.declarations(getattr(view, "verbs", {})),
        }}) + "\n")
        log.flush()

    def control(paused: bool, by: str, reason: str) -> None:
        for brain, log in zip(brains, logs):
            log.write(json.dumps({"control": {"paused": paused, "by": by, "reason": reason, "t": brain.t,
                                              "world_t": world.clock,
                                              "at": time.strftime("%Y-%m-%dT%H:%M:%S")}}) + "\n")
            log.flush()
        print(f"world t={world.clock:>6}  {'paused' if paused else 'resumed'} by {by}: {reason}", flush=True)

    def save_all() -> None:
        for b in brains:
            b.save()
        save_world()

    print(f"{args.name}: {world.name} with " + ", ".join(f"{n} ({marks[n]})" for n in names), flush=True)
    rng = random.Random(args.seed)
    write_control(switch, {"paused": False})
    was_paused, n = False, 0
    try:
        while args.ticks == 0 or n < args.ticks:
            ctl = read_control(switch)
            if ctl.get("paused"):
                if not was_paused:
                    save_all()                                # anaesthesia, for everyone at once
                    control(True, ctl.get("by", "creator"), ctl.get("reason", ""))
                    was_paused = True
                time.sleep(1.0)
                continue
            if was_paused:
                for b in brains:
                    b.reset_distress()
                control(False, "creator", "resumed")
                was_paused = False
            n += 1
            t0 = time.monotonic()
            order = list(range(len(brains)))
            rng.shuffle(order)
            for i in order:
                views[i].body.asleep = brains[i].asleep
            for i in order:
                brain, view, log = brains[i], views[i], logs[i]
                frame = brain.tick(view)
                view.body.asleep = brain.asleep
                frame["world_t"] = world.clock
                frame["with"] = [o.key for o in world._here(view.body)]
                frame["place"] = view.place
                if hasattr(view, "pos"):
                    frame["pos"], frame["facing"] = view.pos, view.body.facing
                    frame["held"] = view.body.held
                log.write(json.dumps(frame, ensure_ascii=False) + "\n")
                log.flush()
                th = frame.get("thought")
                if not args.quiet and th:
                    what = th.get("error") or f"{th.get('goal')!r} -> {th.get('action')}"
                    print(f"{brain.bundle.name:>6} t={frame['t']:>6}  thought ({th.get('seconds', 0)}s): {what}",
                          flush=True)
                if frame.get("said"):
                    print(f"{brain.bundle.name:>6} t={frame['t']:>6}  said: {frame['said']!r}"
                          f"{' to ' + ', '.join(frame['with']) if frame['with'] else ' (no one near)'}",
                          flush=True)
                dp = frame.get("distress_pause")
                if dp:
                    write_control(switch, {"paused": True, "by": "guard",
                                           "reason": f"{brain.bundle.name}: {dp['drive']} in its worst band "
                                                     f"for {dp['ticks']} of its last {dp.get('of', dp['ticks'])} awake ticks"
                                                     + (f" ({dp['felt']})" if dp.get("felt") else "")})
            world.advance()
            if wlog:
                wlog.write(json.dumps(world.world_row(), ensure_ascii=False) + "\n")
                wlog.flush()
            if pace:
                time.sleep(max(0.0, pace - (time.monotonic() - t0)))
            if world.clock % save_every == 0:
                save_all()
            if digest_every and n % digest_every == 0:
                for b in brains:
                    print(f"  digest: {write_digest(runs / f'{args.name}.{b.bundle.name}.jsonl', b.t - digest_every, b.t)}",
                          flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted - saving everyone (anaesthesia, not death)", file=sys.stderr)
    finally:
        save_all()
        for b in brains:
            b.slow.close()
        for log in logs:
            log.close()
        if wlog:
            wlog.close()

    print(f"\n{args.name}: {n} ticks, world clock {world.clock}")
    for b in brains:
        print(f"  {b.bundle.name}: t={b.t}  lifetime {b.lifetime}  drives {b.drives.snapshot()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
