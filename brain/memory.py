"""Hippocampus and consolidated memory (the bundle's episodes and wiki).

- **Episodes** (`episodes.jsonl`): one-shot records of what happened, written
  only when a tick was strong enough (salience + novelty vs. the `write_above`
  knob). Each carries the tick, text, strength, feeling and goal, and whether a
  sleep has consolidated it yet.
- **Recall from partial cues**: the current percepts + goal are matched against
  episodes by word overlap (tf-idf cosine), weighted by strength and recency.
  Episodes still in working memory (the last few ticks) are skipped. Stdlib
  stand-in for embeddings; good enough for small worlds.
- **Wiki** (`wiki/*.md`): what sleep has made of the episodes. Notes are small
  markdown files with a front-matter header; the brain recalls them by the same
  matching and you can read them directly.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path

_STOP = set("""a an the and or but if then of to in on at by for with from into onto as is are was were be
been being it its this that these those i me my mine you your he she they them their we our us
there here so not no do does did done have has had can could would should will just very
now than too also again what which who whom when where why how all any some one""".split())
_WORD = re.compile(r"[a-z][a-z'-]+")


def tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall((text or "").lower()) if w not in _STOP and len(w) > 2]


def slug(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (title or "").lower()).strip("-")
    return s[:60] or "note"


class _Index:
    """Incremental tf-idf over a growing set of documents."""

    def __init__(self):
        self.df: Counter = Counter()
        self.n = 0

    def add(self, toks: list[str]) -> Counter:
        tf = Counter(toks)
        self.df.update(tf.keys())
        self.n += 1
        return tf

    def remove(self, tf: Counter) -> None:
        self.df.subtract(tf.keys())
        self.n -= 1

    def cosine(self, a: Counter, b: Counter) -> float:
        if not a or not b:
            return 0.0

        def w(tf, term):
            return tf[term] * math.log(1.0 + (self.n + 1) / (1.0 + self.df.get(term, 0)))

        common = a.keys() & b.keys()
        if not common:
            return 0.0
        num = sum(w(a, t) * w(b, t) for t in common)
        na = math.sqrt(sum(w(a, t) ** 2 for t in a))
        nb = math.sqrt(sum(w(b, t) ** 2 for t in b))
        return num / (na * nb) if na and nb else 0.0


def _read_note(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    meta, body = {}, text
    if text.startswith("---\n"):
        head, _, body = text[4:].partition("\n---\n")
        for line in head.splitlines():
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    return {"title": meta.get("title", path.stem), "text": body.strip(),
            "updated_t": int(meta.get("updated_t", 0) or 0), "file": path.name}


class Memory:
    def __init__(self, bundle_path, recency_tau: float = 300.0, skip_recent: int = 8,
                 min_similarity: float = 0.0):
        self.root = Path(bundle_path)
        self.ep_path = self.root / "episodes.jsonl"
        self.wiki = self.root / "wiki"
        self.wiki.mkdir(exist_ok=True)
        self.recency_tau = recency_tau
        self.skip_recent = skip_recent
        self.min_similarity = min_similarity      # below this, a cue does not bring anything back
        self.index = _Index()
        self.episodes: list[dict] = []
        self._ep_tf: list[Counter] = []
        if self.ep_path.exists():
            with open(self.ep_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self._index_episode(json.loads(line))
        self.notes: dict[str, dict] = {}
        self._note_tf: dict[str, Counter] = {}
        for p in sorted(self.wiki.glob("*.md")):
            if p.name != "index.md":
                self._index_note(_read_note(p))

    # -- episodes ------------------------------------------------------------
    def _index_episode(self, ep: dict) -> None:
        self.episodes.append(ep)
        self._ep_tf.append(self.index.add(tokens(ep["text"])))

    def write(self, ep: dict) -> dict:
        ep = dict(ep, id=len(self.episodes), consolidated=False)
        with open(self.ep_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")
        self._index_episode(ep)
        return ep

    def unconsolidated(self, limit: int) -> list[dict]:
        eps = [e for e in self.episodes if not e.get("consolidated")]
        if len(eps) > limit:                       # keep the strongest, in time order
            keep = {e["id"] for e in sorted(eps, key=lambda e: -e.get("strength", 0))[:limit]}
            eps = [e for e in eps if e["id"] in keep]
        return eps

    def mark_consolidated(self, upto_id: int) -> None:
        """A sleep closes out every episode up to `upto_id`: the replayed ones were
        consolidated, the weak ones left out of the replay are never transferred."""
        for e in self.episodes:
            if e["id"] <= upto_id:
                e["consolidated"] = True
        tmp = self.ep_path.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for e in self.episodes:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        os.replace(tmp, self.ep_path)

    def recall(self, cue: str, t_now: int, k: int = 1) -> list[dict]:
        q = Counter(tokens(cue))
        scored = []
        for ep, tf in zip(self.episodes, self._ep_tf):
            if t_now - ep["t"] <= self.skip_recent:
                continue
            sim = self.index.cosine(q, tf)
            if sim <= self.min_similarity:
                continue
            recency = math.exp(-(t_now - ep["t"]) / self.recency_tau)
            scored.append((sim * (0.5 + ep.get("strength", 0.5)) * (0.3 + 0.7 * recency), ep))
        scored.sort(key=lambda x: -x[0])
        return [ep for _, ep in scored[:k]]

    # -- wiki ------------------------------------------------------------------
    def _index_note(self, note: dict) -> None:
        key = slug(note["title"])
        if key in self._note_tf:
            self.index.remove(self._note_tf[key])
        self.notes[key] = note
        self._note_tf[key] = self.index.add(tokens(note["title"] + " " + note["text"]))

    def knowledge(self, cue: str, k: int = 1) -> list[dict]:
        q = Counter(tokens(cue))
        scored = [(self.index.cosine(q, tf), self.notes[key]) for key, tf in self._note_tf.items()]
        return [n for s, n in sorted(scored, key=lambda x: -x[0]) if s > self.min_similarity][:k]

    def write_notes(self, notes: list[dict], t: int) -> tuple[int, int]:
        created = updated = 0
        for n in notes:
            title, text = str(n.get("title", "")).strip(), str(n.get("text", "")).strip()
            if not title or not text:
                continue
            key = slug(title)
            if key in self.notes:
                updated += 1
            else:
                created += 1
            path = self.wiki / f"{key}.md"
            path.write_text(f"---\ntitle: {title}\nupdated_t: {t}\n---\n{text}\n", encoding="utf-8")
            self._index_note({"title": title, "text": text, "updated_t": t, "file": path.name})
        lines = ["# What this mind knows", ""]
        lines += [f"- [{n['title']}]({n['file']}) — t={n['updated_t']}"
                  for n in sorted(self.notes.values(), key=lambda n: n["title"].lower())]
        (self.wiki / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return created, updated

    def all_notes(self) -> list[dict]:
        return sorted(self.notes.values(), key=lambda n: n["title"].lower())
