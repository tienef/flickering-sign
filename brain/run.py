"""Run a brain bundle in a world and write a JSONL run log.

    python -m brain.run --bundle bundles/ada --ticks 300 --out runs/brain-ada.jsonl

The bundle is created from `brain/template/` if it does not exist, and resumed
if it does: stop a run (Ctrl-C) and start it again, and the same self carries
on from `state.json` — possibly in a different world.

**World time:** one tick is one second of world time (`loop.seconds_per_tick`,
the default `--pace`); `--ticks 0` runs until stopped. **Pause:** the run polls
`<log>.control.json` ({"paused": true}) — written by the observer or by the
distress guard — and while paused nothing ticks, world included (anaesthesia);
each pause and resume is a `{"control": …}` line in the log. **Digests:** every
`loop.digest_every` ticks (a world day) a markdown digest is written next to
the log (`brain/digest.py`).
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from collections import Counter
from pathlib import Path

from .backends import make_fast, make_slow
from .brain import Brain
from .bundle import Bundle
from .digest import write_digest
from .worlds import make_world


def _control_path(out: Path) -> Path:
    return out.with_name(out.stem + ".control.json")


def read_control(out: Path) -> dict:
    try:
        return json.loads(_control_path(out).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_control(out: Path, state: dict) -> None:
    p = _control_path(out)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    tmp.replace(p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle", default="bundles/ada")
    ap.add_argument("--world", default="toy")
    ap.add_argument("--ticks", type=int, default=300, help="ticks to run; 0 = until stopped")
    ap.add_argument("--fast", default="stub", help="System 1 backend: stub | laya")
    ap.add_argument("--slow", default="stub", help="System 2 backend: stub | qwen (any OpenAI-compatible endpoint; alias: openai)")
    ap.add_argument("--fast-device", default=None,
                    help="override the bundle's fast.device (cpu | cuda)")
    ap.add_argument("--think-budget", type=int, default=None,
                    help="override the bundle's slow.think_budget (thinking tokens per request)")
    ap.add_argument("--model", default=None,
                    help="override the bundle's slow.model (the name your endpoint gives the model)")
    ap.add_argument("--clamp", action="append", default=[], metavar="DIAL=LEVEL",
                    help="fix a dial for this run, e.g. --clamp noradrenaline=0.9 (repeatable)")
    ap.add_argument("--pace", type=float, default=None,
                    help="minimum wall seconds per tick (default: the bundle's loop.seconds_per_tick; "
                         "0 = as fast as possible)")
    ap.add_argument("--overlay", default=None,
                    help="genome overlay applied when the bundle is CREATED (brain/variants/<name>.json)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="JSONL run log (default runs/brain-<name>.jsonl)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    # `kill` (SIGTERM) saves like Ctrl-C: nohup'd background runs ignore SIGINT
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    bundle = Bundle.open(args.bundle, overlay=args.overlay)
    world = make_world(args.world, seed=args.seed, bundle=bundle.path)
    slow_cfg = dict(bundle.config.get("slow", {}))
    if args.model:
        slow_cfg["model"] = args.model
    if args.think_budget is not None:
        slow_cfg["think_budget"] = args.think_budget
    fast_cfg = dict(bundle.config.get("fast", {}))
    if args.fast_device:
        fast_cfg["device"] = args.fast_device
    clamps = dict((k, float(v)) for k, v in (c.split("=", 1) for c in args.clamp))
    slow = make_slow(args.slow, seed=args.seed, cfg=slow_cfg)
    if args.fast == "laya" and fast_cfg.get("device", "cpu") != "cpu" and hasattr(slow, "warm"):
        # Swap the card to System 2's model first, so Laya lands beside the engine
        # that leaves it room (the q27 brain profile), never in another's slack.
        print(f"warming {slow.model} (may swap the card) ...", flush=True)
        print(f"  ready in {slow.warm()} s", flush=True)
    brain = Brain(bundle, make_fast(args.fast, seed=args.seed, cfg=fast_cfg), slow, clamps=clamps)
    out = Path(args.out or f"runs/brain-{bundle.name}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    save_every = int(bundle.config.get("loop", {}).get("save_every", 25))
    pace = args.pace if args.pace is not None else float(brain.loop.get("seconds_per_tick", 1.0))
    digest_every = int(brain.loop.get("digest_every", 0))
    resumed_at = brain.t

    actions, escalations, thoughts = Counter(), 0, 0
    with open(out, "a", encoding="utf-8") as log:
        log.write(json.dumps({"meta": {
            "bundle": bundle.name, "world": world.name, "fast": brain.fast.name,
            "slow": brain.slow.name, "seed": args.seed, "resumed_at": resumed_at,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "clamps": clamps,
            "variant": bundle.config.get("variant"), "pace": pace,
        }}) + "\n")

        def control(paused: bool, by: str, reason: str) -> None:
            state = {"paused": paused, "by": by, "reason": reason, "t": brain.t,
                     "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
            log.write(json.dumps({"control": state}) + "\n")
            log.flush()
            if not args.quiet or by == "guard":
                print(f"t={brain.t:>5}  {'paused' if paused else 'resumed'} by {by}: {reason}", flush=True)

        write_control(out, {"paused": False})
        was_paused = False
        try:
            n = 0
            while args.ticks == 0 or n < args.ticks:
                ctl = read_control(out)
                if ctl.get("paused"):
                    if not was_paused:
                        brain.save()                         # anaesthesia: the self is on disk
                        control(True, ctl.get("by", "creator"), ctl.get("reason", ""))
                        was_paused = True
                    time.sleep(1.0)
                    continue
                if was_paused:
                    brain.reset_distress()
                    control(False, "creator", "resumed")
                    was_paused = False
                n += 1
                t0 = time.monotonic()
                frame = brain.tick(world)
                log.write(json.dumps(frame, ensure_ascii=False) + "\n")
                log.flush()
                actions[frame["action"]] += 1
                escalations += bool(frame.get("escalated"))
                th = frame.get("thought")
                thoughts += bool(th and not th.get("error"))
                if not args.quiet and th:
                    what = th.get("error") or f"{th.get('goal')!r} -> {th.get('action')}"
                    print(f"t={frame['t']:>5}  thought ({th.get('seconds', 0)}s, "
                          f"{th.get('tokens', 0)} tok): {what}", flush=True)
                c = frame.get("consolidation")
                if not args.quiet and c:
                    what = c.get("error") or (f"+{c['created']} notes, {c['updated']} revised, "
                                              f"{len(c['contradictions'])} contradictions, "
                                              f"{len(c['resolved'])} resolved: {c['notes']}")
                    print(f"t={frame['t']:>5}  slept on it ({c.get('seconds')}s, "
                          f"{c.get('tokens')} tok): {what}", flush=True)
                dp = frame.get("distress_pause")
                if dp:
                    write_control(out, {"paused": True, "by": "guard",
                                        "reason": f"{dp['drive']} in its worst band for {dp['ticks']} of its last {dp.get('of', dp['ticks'])} awake ticks"
                                                  + (f" ({dp['felt']})" if dp.get("felt") else "")})
                if frame.get("asleep") and brain.consolidating:
                    time.sleep(float(brain.loop.get("sleep_wait_seconds", 0.0)))   # dreaming takes wall time
                elif pace:
                    time.sleep(max(0.0, pace - (time.monotonic() - t0)))
                if brain.t % save_every == 0:
                    brain.save()
                if digest_every and brain.t % digest_every == 0:
                    print(f"  digest: {write_digest(out, brain.t - digest_every, brain.t)}", flush=True)
        except KeyboardInterrupt:
            print("\ninterrupted - saving state (anaesthesia, not death)", file=sys.stderr)
        finally:
            brain.save()
            brain.slow.close()

    ran = brain.t - resumed_at
    print(f"\n{bundle.name}: ticks {resumed_at}->{brain.t} ({ran} this run) in {world.name}")
    print(f"  actions: {dict(actions.most_common())}")
    print(f"  escalated to System 2: {escalations}   thoughts landed: {thoughts}")
    print(f"  lifetime: {brain.lifetime}")
    print(f"  memory: {len(brain.memory.episodes)} episodes, {len(brain.memory.notes)} wiki notes, "
          f"{len(brain.open_contradictions)} open contradictions")
    print(f"  drives [level, urgency]: {brain.drives.snapshot()}")
    print(f"  dials: {brain.dials.snapshot()}")
    if getattr(brain.fast, "calls", 0):
        print(f"  System 1: {brain.fast.calls} passes, "
              f"{brain.fast.seconds / brain.fast.calls:.2f} s/pass")
    print(f"  log: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
