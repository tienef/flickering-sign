"""The bundle: a brain is a folder.

    bundles/<name>/
      brain.json      identity, knobs, loop settings
      drives.json     drive declarations
      dials.json      dial declarations
      state.json      tick, drive/dial levels, goal line, familiarity
      episodes.jsonl  hippocampus (step 4)
      wiki/           consolidated knowledge (step 4)

A new bundle is copied from `brain/template/`, optionally with a genome overlay
(`brain/variants/*.json`: {"brain.json": {...}, "drives.json": {...}}, merged key
by key, lists replaced) — how an A/B arm differs from the template, in one file.
Config files are the brain's "genome" and are only read; `state.json` is its
current self and is written atomically, so a crash or a box power-off never
leaves half a self.
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

TEMPLATE = Path(__file__).parent / "template"


def _read(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


class Bundle:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.config = _read(self.path / "brain.json")
        self.drives = _read(self.path / "drives.json")
        self.dials = _read(self.path / "dials.json")
        sp = self.path / "state.json"
        self.state = _read(sp) if sp.exists() else {}

    @property
    def name(self) -> str:
        return self.config.get("name", self.path.name)

    @classmethod
    def open(cls, path, *, create: bool = True, overlay: str | None = None) -> "Bundle":
        path = Path(path)
        if not (path / "brain.json").exists():
            if not create:
                raise FileNotFoundError(f"no bundle at {path}")
            shutil.copytree(TEMPLATE, path, dirs_exist_ok=True)
            overlays = [o for o in (overlay or "").split(",") if o]      # several, applied in order
            for ov in overlays:
                for fname, over in _read(Path(ov)).items():
                    if fname.startswith("_"):
                        continue
                    merged = _merge(_read(path / fname), over)
                    with open(path / fname, "w", encoding="utf-8") as f:
                        json.dump(merged, f, indent=2, ensure_ascii=False)
            cfg = _read(path / "brain.json")
            cfg["name"] = path.name
            cfg["born"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            if overlay:
                cfg["variant"] = "+".join(Path(o).stem for o in overlays)
            with open(path / "brain.json", "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
            (path / "wiki").mkdir(exist_ok=True)
        return cls(path)

    def save_state(self, state: dict) -> None:
        self.state = state
        tmp = self.path / "state.json.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
        os.replace(tmp, self.path / "state.json")
