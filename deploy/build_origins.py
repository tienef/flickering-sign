"""Build the archive's static corpus (brain/origins/corpus.json) from the repo.

    python deploy/build_origins.py

Run it before a run in the `valley` world (the archive reads it). Papers, in shelf order:
README.md, BRAIN.md and PLAN.md cut at their headings, each module's docstring,
and the git history (dates and subjects only: no authors). Every line goes
through `brain.origins.clean` (secrets, addresses, paths, names, and the blind).
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from brain.origins import _BLIND, CORPUS, paper  # noqa: E402

MODULES = ["events", "drives", "dials", "backends", "worlds", "brain", "bundle", "memory",
           "predictor", "origins", "report", "run"]


def sections(path: Path, prefix: str) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"(?m)^(#{1,3} .+)$", text)
    out, title = [], prefix
    if parts[0].strip():
        out.append(paper(prefix, parts[0]))
    for i in range(1, len(parts), 2):
        heading = parts[i].lstrip("#").strip()
        if _BLIND.search(heading):
            continue                     # a section about what is being scored is withheld whole
        out.append(paper(f"{prefix}: {heading}", parts[i + 1]))
    return [p for p in out if p]


def main() -> int:
    papers = [paper("README.md", (ROOT / "README.md").read_text(encoding="utf-8"))]
    papers += sections(ROOT / "brain" / "BRAIN.md", "BRAIN.md")
    for m in MODULES:
        src = (ROOT / "brain" / f"{m}.py").read_text(encoding="utf-8")
        doc = ast.get_docstring(ast.parse(src))
        if doc:
            papers.append(paper(f"brain/{m}.py", doc))
    try:
        log = subprocess.run(["git", "log", "--reverse", "--format=%ad %s", "--date=short"],
                             cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True).stdout
        papers.append(paper("git log (the history of the project, oldest first)", log))
    except (OSError, subprocess.CalledProcessError):
        print("no git history here: the archive will hold no git log")
    papers += sections(ROOT / "brain" / "PLAN.md", "PLAN.md")
    papers = [p for p in papers if p]
    CORPUS.parent.mkdir(exist_ok=True)
    CORPUS.write_text(json.dumps(papers, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{CORPUS.relative_to(ROOT)}: {len(papers)} papers, {sum(len(p['pages']) for p in papers)} pages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
