"""What the archive holds: the brain's origins, read-only and sanitised.

Two sources, turned into papers of short pages:
- **static** — `brain/origins/corpus.json`, built on the PC at deploy time by
  `deploy/build_origins.py` from the repo: the README, the design (BRAIN.md),
  the order of work (PLAN.md), each module's docstring and the git history;
- **live** — the brain's own bundle, read when the world is built: its genome
  (`brain.json`, `drives.json`), the index of its wiki and its first memories.

Nothing is executed and nothing is written. Every line passes `clean()`, which
drops anything that looks like a secret, an address, a machine path or a
person's name: the archive is a sanitised copy, never the live filesystem.
It also drops the lines that name what the experiment hopes to see (curiosity,
the checklist, questions about origins): the mind may read how it was made, but
not what we are watching for — a blind, as in a trial. The gaps are visible.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

CORPUS = Path(__file__).parent / "origins" / "corpus.json"
PAGE_CHARS = 280

_SECRET = re.compile(
    r"(?i)(api[_ -]?key|\bkey\b.*=|token|password|secret|\.env\b|tailnet|tailscale|"
    r"\b\d{1,3}(?:\.\d{1,3}){3}\b|/home/|/mnt/|~/|\b[a-z]:[\\/]|@[a-z0-9-]+\.[a-z]|"
    r"gmail|boxctl_token|brain_gateway)")
# Private words (the creator's names, usernames, machine names) kept out of the
# archive: comma-separated in BRAIN_PRIVATE_WORDS, never written in the source.
_PRIVATE = [w.strip() for w in os.environ.get("BRAIN_PRIVATE_WORDS", "").split(",") if w.strip()]
_PRIVATE_RE = re.compile("(?i)" + "|".join(map(re.escape, _PRIVATE))) if _PRIVATE else None


# the blind: lines naming the behaviour being scored
_BLIND = re.compile(r"(?i)(curio|checklist|emerg|origins|questions? about (?:it|them)self|"
                    r"seeks? information|information-seeking|asks? questions|questions on its own|"
                    r"self-generated|learning frontier|where it is actually learning)")


def clean(text: str) -> str:
    """Drop every line that could leak a secret, an address, a path or a name, or
    that names what the experiment is watching for."""
    return "\n".join(ln for ln in text.splitlines()
                     if not _SECRET.search(ln) and not (_PRIVATE_RE and _PRIVATE_RE.search(ln))
                     and not _BLIND.search(ln)).strip()


def paginate(text: str, size: int = PAGE_CHARS) -> list[str]:
    words, pages, cur = " ".join(text.split()).split(" "), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > size:
            pages.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        pages.append(cur)
    return pages


def paper(title: str, text: str) -> dict | None:
    pages = paginate(clean(text))
    return {"title": title, "pages": pages} if pages else None


def _live(bundle: Path) -> list[dict]:
    out = []
    bj = bundle / "brain.json"
    if bj.exists():
        cfg = json.loads(bj.read_text(encoding="utf-8"))
        cfg.get("slow", {}).pop("env_file", None)
        out.append(paper("brain.json (the genome of the mind named "
                         f"'{cfg.get('name', bundle.name)}')", json.dumps(cfg, indent=1, ensure_ascii=False)))
    dj = bundle / "drives.json"
    if dj.exists():
        out.append(paper("drives.json", dj.read_text(encoding="utf-8")))
    idx = bundle / "wiki" / "index.md"
    if idx.exists():
        out.append(paper("wiki/index.md (the notes this mind keeps)", idx.read_text(encoding="utf-8")))
    ep = bundle / "episodes.jsonl"
    if ep.exists():
        first = []
        with open(ep, encoding="utf-8") as f:
            for line, _ in zip(f, range(6)):
                e = json.loads(line)
                first.append(f"(t={e.get('t')}) {e.get('text', '')}")
        out.append(paper("episodes.jsonl (this mind's first memories)", "\n".join(first)))
    return [p for p in out if p]


def load_archive(bundle: str | Path | None) -> list[dict]:
    """The papers on the archive's shelves, in shelf order."""
    static = json.loads(CORPUS.read_text(encoding="utf-8")) if CORPUS.exists() else []
    live = _live(Path(bundle)) if bundle else []
    # shelf order: what the project is, then this mind's own files, then the rest
    return static[:1] + live + static[1:]
