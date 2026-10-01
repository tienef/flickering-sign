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
- **Selection of notes (step 10b):** a note is a hypothesis. When the genome has a
  review, each sleep first checks every note against the day's experiences
  (`h` held, `f` failed; untested days leave no mark) and keeps the last verdicts
  in the note's `tests`. A note failed at least `retire_after` times in that window,
  and more often than it held, is retired to `wiki/retired/` (kept, never recalled
  again). Recall favours notes that have held: fitness = (held + 1) / (tested + 2),
  so an untested note (0.5) ranks exactly as before.
- **Cues written at sleep (recall_index.py):** a note may carry, in its front matter, the
  moments it will matter (`cue: <when> => <recall>`, written by System 2 at sleep) and when
  they were written (`cued_t`). With a cue index attached, notes are recalled through it
  instead of the word overlap (episodes keep the word overlap). A cue may end with the
  body's action by which the mind did what its line says (`cue: <when> => <recall> => <act>`,
  v4), named by the sleep from the mind's own memories.
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
    meta, body, cues = {}, text, []
    if text.startswith("---\n"):
        head, _, body = text[4:].partition("\n---\n")
        for line in head.splitlines():
            k, _, v = line.partition(":")
            if k.strip() == "cue":
                when, _, rest = v.partition("=>")
                recall, _, act = rest.partition("=>")
                if when.strip() and recall.strip():
                    cues.append({"when": when.strip(), "recall": recall.strip(),
                                 **({"act": act.strip()} if act.strip() else {})})
            else:
                meta[k.strip()] = v.strip()
    note = {"title": meta.get("title", path.stem), "text": body.strip(),
            "updated_t": int(meta.get("updated_t", 0) or 0), "file": path.name,
            "tests": meta.get("tests", "")}
    if cues or meta.get("cued_t"):
        note.update(cues=cues, cued_t=int(meta.get("cued_t") or 0))
    return note


def _one_line(text, limit: int = 240) -> str:
    """A cue's field on one front-matter line: no newlines, no `=>` (the separator)."""
    return re.sub(r"\s+", " ", str(text or "")).replace("=>", "->").strip()[:limit]


def fitness(note: dict) -> float:
    """How well a note has held up: (held + 1) / (tested + 2); 0.5 when never tested."""
    t = note.get("tests", "")
    return (t.count("h") + 1) / (len(t) + 2)


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
        self.cue_index = None                     # a recall_index.CueIndex, attached by the brain
        self.episodes: list[dict] = []
        self._ep_tf: list[Counter] = []
        if self.ep_path.exists():
            with open(self.ep_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self._index_episode(json.loads(line))
        self.notes: dict[str, dict] = {}
        # traces written awake at a strong moment (genome `traces`): recalled like notes until the next
        # rebuild of the cue index (after the night's consolidation, which saw them), never in the wiki
        self.provisional: dict[str, dict] = {}
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

    def knowledge(self, cue: str, k: int = 1, query: str | None = None) -> list[dict]:
        """The notes this cue brings back. With a cue index, `query` (else the cue) is matched against
        the cues written at sleep, and each note comes back with the `recall` line to read."""
        if self.cue_index is not None:
            return self.cue_index.search(query or cue, {**self.notes, **self.provisional}, k)
        q = Counter(tokens(cue))
        scored = [(self.index.cosine(q, tf), self.notes[key]) for key, tf in self._note_tf.items()]
        scored = [(sim * 2 * fitness(n), n) for sim, n in scored if sim > self.min_similarity]
        return [n for s, n in sorted(scored, key=lambda x: -x[0])][:k]

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
            # a revised note keeps its record: a lineage, not a fresh hypothesis; its cues stay
            # until the night's cue writing replaces them (it asks again for every note revised)
            old = self.notes.get(key, {})
            self._save_note({"title": title, "text": text, "updated_t": t, "file": f"{key}.md",
                             "tests": old.get("tests", ""),
                             **({"cues": old["cues"], "cued_t": old["cued_t"]} if "cued_t" in old else {})})
        self._write_index()
        return created, updated

    @staticmethod
    def _cue_head(note: dict) -> str:
        if "cued_t" not in note:
            return ""
        return f"cued_t: {note['cued_t']}\n" + "".join(
            f"cue: {c['when']} => {c['recall']}" + (f" => {c['act']}" if c.get("act") else "") + "\n"
            for c in note.get("cues") or [])

    def _save_note(self, note: dict) -> None:
        tests = f"tests: {note['tests']}\n" if note.get("tests") else ""
        (self.wiki / note["file"]).write_text(
            f"---\ntitle: {note['title']}\nupdated_t: {note['updated_t']}\n{tests}{self._cue_head(note)}---\n"
            f"{note['text']}\n", encoding="utf-8", newline="\n")
        self._index_note(note)

    # -- cues written at sleep (recall_index.py) -------------------------------
    def needs_cues(self) -> list[dict]:
        """The notes this sleep touched (created or revised since their cues were written), and any
        never cued (a failed night's are asked again)."""
        return [n for n in self.all_notes() if "cued_t" not in n or n["updated_t"] > n["cued_t"]]

    def write_cues(self, asked: list[str], cues: list[dict], t: int, acts=None) -> dict:
        """The night's cues replace those of the notes asked (`asked`: titles); a note asked and given no
        moment keeps none (it never matters to what the mind does). Titles are matched by slug, then
        case-insensitively. With `acts` (the body's actions, v4), a cue keeps its `act` only if it is one
        of them ("none" or anything else: no act). Returns counts for the log."""
        by_lower = {n["title"].lower().strip(): key for key, n in self.notes.items()}
        new: dict[str, list] = {slug(a): [] for a in asked if slug(a) in self.notes}
        lost = bad = 0
        for c in cues:
            title = str(c.get("title") or "")
            key = slug(title) if slug(title) in new else by_lower.get(title.lower().strip())
            if key not in new:
                lost += 1
                continue
            when, recall = _one_line(c.get("when")), _one_line(c.get("recall"))
            if not (when and recall):
                continue
            cue = {"when": when, "recall": recall}
            if acts is not None:
                act = _one_line(c.get("act")).lower().rstrip(".")
                if act in acts:
                    cue["act"] = act
                elif act and act != "none":
                    bad += 1                       # not one of the body's actions: the cue keeps none
            new[key].append(cue)
        for key, cs in new.items():
            self.notes[key].update(cues=cs, cued_t=t)
            self._save_note(self.notes[key])
        out = {"asked": len(new), "cues": sum(len(c) for c in new.values()),
               "with_cues": sum(1 for c in new.values() if c), "lost": lost}
        if acts is not None:
            out.update(with_act=sum(1 for cs in new.values() for c in cs if c.get("act")), bad_act=bad)
        return out

    def rebuild_index(self) -> int | None:
        """The cue index, rebuilt in full from the wiki as it is now (None without one); the day's
        provisional traces go with it."""
        self.provisional = {}
        return self.cue_index.rebuild(self.notes) if self.cue_index is not None else None

    def add_trace(self, key: str, when: str, recall: str, act: str | None, t: int) -> bool:
        """A provisional trace (genome `traces`): a cue unit in the index at once and a note-like entry
        for what comes back with it. False when the same trace is already there or there is no index."""
        if self.cue_index is None or any(n["cues"][0]["when"] == when and n["cues"][0]["recall"] == recall
                                         for n in self.provisional.values()):
            return False
        self.provisional[key] = {"title": f"a moment at t={t}", "text": recall, "provisional": True, "t": t,
                                 "cues": [{"when": when, "recall": recall, "act": act}]}
        self.cue_index.add(key, when, recall, act)
        return True

    def _write_index(self) -> None:
        lines = ["# What this mind knows", ""]
        lines += [f"- [{n['title']}]({n['file']}) — t={n['updated_t']}"
                  for n in sorted(self.notes.values(), key=lambda n: n["title"].lower())]
        (self.wiki / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def review(self, verdicts: dict[str, str], t: int, window: int = 6,
               retire_after: int = 2) -> dict:
        """Apply a sleep's verdicts ({title: held | failed | untested}) and retire the
        notes that keep failing. Returns what happened, for the log."""
        out = {"held": [], "failed": [], "retired": []}
        for title, verdict in verdicts.items():
            key = slug(title)
            note = self.notes.get(key)
            if note is None or verdict not in ("held", "failed"):
                continue
            note["tests"] = (note.get("tests", "") + verdict[0])[-window:]
            out[verdict].append(note["title"])
            fails = note["tests"].count("f")
            if fails >= retire_after and fails > note["tests"].count("h"):
                self.retire(key, t)
                out["retired"].append(note["title"])
            else:
                self._save_note(note)
        self._write_index()
        return out

    def retire(self, key: str, t: int) -> None:
        note = self.notes.pop(key)
        self.index.remove(self._note_tf.pop(key))
        gone = self.wiki / "retired"
        gone.mkdir(exist_ok=True)
        (gone / note["file"]).write_text(
            f"---\ntitle: {note['title']}\nupdated_t: {note['updated_t']}\nretired_t: {t}\n"
            f"tests: {note.get('tests', '')}\n{self._cue_head(note)}---\n{note['text']}\n",
            encoding="utf-8", newline="\n")
        (self.wiki / note["file"]).unlink(missing_ok=True)

    def all_notes(self) -> list[dict]:
        return sorted(self.notes.values(), key=lambda n: n["title"].lower())
