"""Recall by cues written at sleep: a vector index over the wiki's cues (PLAN "A vector index built at sleep").

The wiki stays the source of truth. At each sleep, after the review and the consolidation, System 2 writes for
each note it touched the moments it will matter (`when`: what the mind then senses) and one line to recall then
(`recall`); they are kept in the note's front matter (`cue: <when> => <recall>`). This index is a by-product of
the wiki, thrown away and rebuilt in full at every sleep (and when the brain starts): one unit per cue, its `when`
line embedded. Neither system reads vectors: the index only chooses the text that reaches them, the `recall` line
for Laya's `knowledge` line and the whole note for System 2's "What the mind knows".

Recall is `cue-near` (the offline bench's best): the query is what is in front, at my feet, carried or held (the
percepts starting with the genome's `near` prefixes; the whole situation + goal when none), scored by cosine
against the `when` lines, one unit per note (its best), no fitness in the score (bge cosines are compressed,
~0.5-0.8: fitness would swamp them), above `min_similarity` (set for ~10% of awake ticks with a note).

Genome: loop.recall_index = {"kind": "cues", "model": "BAAI/bge-small-en-v1.5", "device": "cuda",
"min_similarity": 0.722, "near": [...], "habituate_above": 0.15}; model "hash" is a stdlib-and-numpy stand-in for the
stub smoke.

Habituation (`habituate_above`, after cues2): a cue whose `when` line came back on more than that share of the awake
moments since the last rebuild tells nothing about the moment ("Bread is at my feet" above the threshold in front of
a bare floor) and is left out of the index at the rebuild; the cue overload principle (Watkins & Watkins 1975: a cue
loses its power with the number of things it is bound to) and habituation to a stimulus that is always there. It
stays in the wiki (the note's front matter): the next rebuild counts it again.

The cue's act (v4, PLAN "v4 cues"; owner OK 2026-09-29): a cue may carry the body's action by which the mind did what
its line says, named by the sleep from the mind's own memories ("What my acts did"), or none. It rides with the unit
and comes back with the note (`act`); the brain decides what to do with it (loop.recall_index.laya_line, the
`cue_act_pull` knob). Nothing here knows any action's name.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np

from .memory import tokens

_EMBEDDERS: dict = {}


class _Bge:
    """bge: CLS pooling, normalised; queries get the model's retrieval instruction (as probe/vector_recall.py)."""

    def __init__(self, model: str, device: str):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).to(device).eval()
        if device == "cuda":
            self.model.half()
        self.qp = "Represent this sentence for searching relevant passages: " if "bge" in model.lower() else ""
        self.name = f"{model}@{device}"

    def encode(self, texts: list[str], query: bool = False) -> np.ndarray:
        torch, out = self.torch, []
        for i in range(0, len(texts), 256):
            b = self.tok([(self.qp if query else "") + t for t in texts[i:i + 256]], padding=True,
                         truncation=True, max_length=256, return_tensors="pt").to(self.device)
            with torch.no_grad():
                v = self.model(**b).last_hidden_state[:, 0].float()
            out.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 384), dtype=np.float32)


class _Hash:
    """Hashed bag of words, normalised: only for the stub smoke (no torch)."""

    name = "hash"

    def encode(self, texts: list[str], query: bool = False) -> np.ndarray:
        m = np.zeros((len(texts), 256), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in tokens(t):
                m[i, int(hashlib.md5(w.encode()).hexdigest()[:6], 16) % 256] += 1.0
        n = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.where(n > 0, n, 1.0)


def embedder(model: str, device: str):
    """One embedder per (model, device) in the process: both minds of a pair share it."""
    key = (model, device)
    if key not in _EMBEDDERS:
        _EMBEDDERS[key] = _Hash() if model == "hash" else _Bge(model, device)
    return _EMBEDDERS[key]


class CueIndex:
    def __init__(self, cfg: dict):
        self.emb = embedder(cfg.get("model", "BAAI/bge-small-en-v1.5"), cfg.get("device", "cuda"))
        self.min_similarity = float(cfg.get("min_similarity", 0.722))
        self.near = tuple(cfg.get("near", ("In front of me:", "I carry", "I hold", "At my feet")))
        self.habituate_above = cfg.get("habituate_above")      # None: no habituation
        self.heard: Counter = Counter()                        # the awake moments' queries since the last rebuild
        self.habituated = 0
        self.units: list[tuple[str, str, str, str | None]] = []   # (note key, when, recall, act)
        self.U = np.zeros((0, 1), dtype=np.float32)
        self._q: dict[str, np.ndarray] = {}                    # query cache (states repeat a lot)

    def rebuild(self, notes: dict[str, dict]) -> int:
        """The whole index from the wiki's notes as they are now; retired notes are not in `notes`."""
        self.units = [(key, c["when"], c["recall"], c.get("act")) for key, n in notes.items()
                      for c in n.get("cues") or []]
        self.U = self.emb.encode([u[1] for u in self.units])
        self.habituated = 0
        if self.habituate_above is not None and self.units and self.heard:
            qs = list(self.heard)
            w = np.array([self.heard[q] for q in qs], dtype=np.float32)
            rate = ((self._encode_q(qs) @ self.U.T > self.min_similarity) * w[:, None]).sum(0) / w.sum()
            keep = rate <= float(self.habituate_above)
            self.habituated = int((~keep).sum())
            self.units = [u for u, k in zip(self.units, keep) if k]
            self.U = self.U[keep]
        self.heard.clear()
        return len(self.units)

    def add(self, key: str, when: str, recall: str, act: str | None) -> None:
        """One unit added now (a provisional trace, written awake); the next rebuild drops it."""
        v = self.emb.encode([when])
        self.U = v if not self.units else np.vstack([self.U, v])
        self.units.append((key, when, recall, act))

    def hear(self, query: str) -> None:
        """An awake moment's query, counted for the habituation at the next rebuild."""
        if self.habituate_above is not None and query.strip():
            self.heard[query] += 1

    def _encode_q(self, qs: list[str]) -> np.ndarray:
        todo = [q for q in qs if q not in self._q]
        if todo:
            if len(self._q) > 20000:
                self._q.clear()
            for q, v in zip(todo, self.emb.encode(todo, query=True)):
                self._q[q] = v
        return np.stack([self._q[q] for q in qs])

    def query(self, percepts: list[str], cue: str) -> str:
        """cue-near: what is in front / at my feet / carried / held; the whole cue when nothing is near."""
        return " ".join(p for p in percepts if p.startswith(self.near)) or cue.strip()

    def search(self, query: str, notes: dict[str, dict], k: int = 1) -> list[dict]:
        if not self.units or not query.strip():
            return []
        q = self._encode_q([query])[0]
        sims = self.U @ q
        best: dict[str, tuple[float, str, str | None]] = {}    # one unit per note: its best
        for j in np.argsort(-sims):
            s = float(sims[j])
            if s <= self.min_similarity or len(best) == k:
                break
            key = self.units[j][0]
            if key in notes and key not in best:
                best[key] = (s, self.units[j][2], self.units[j][3])
        return [dict(notes[key], recall=line, act=act, cue_sim=round(s, 4))
                for key, (s, line, act) in best.items()]
