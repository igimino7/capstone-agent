"""Shared helpers for the code-retriever skill (ME/CE/AM 295, Lab 5).

Layout assumed (repository root = the capstone-agent repo):
    corpus/manifest.json      one entry per document
    corpus/<files>            PDFs (or .txt / .md)
    index/                    written by build_index.py, read by everything else
    .claude/skills/code-retriever/   this folder

Override the root with the environment variable RETRIEVER_ROOT if needed.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("RETRIEVER_ROOT", SKILL_DIR.parents[2])).resolve()
CORPUS = ROOT / "corpus"
INDEX = ROOT / "index"
MANIFEST = CORPUS / "manifest.json"

DEFAULT_ST_MODEL = "BAAI/bge-small-en-v1.5"
REQUIRED_FIELDS = ("name", "edition", "source", "licence", "file")
SUPPORTED = (".pdf", ".txt", ".md")


def die(msg: str, code: int = 2) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


# ---------------------------------------------------------------- manifest
def load_manifest() -> list[dict]:
    if not MANIFEST.exists():
        die(f"{MANIFEST.relative_to(ROOT)} not found — write it first (Lab 5, Part A).")
    try:
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        die(f"corpus/manifest.json is not valid JSON: {e}")
    if isinstance(data, dict):
        data = data.get("documents", [data])
    if not isinstance(data, list):
        die("corpus/manifest.json must be a list of entries (or {\"documents\": [...]})")
    return data


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- text normalisation
_DASHES = dict.fromkeys(map(ord, "‐‑‒–—−"), "-")
_QUOTES = {ord("‘"): "'", ord("’"): "'", ord("“"): '"', ord("”"): '"'}


def norm(text: str) -> str:
    """Normalise for verbatim comparison: unicode NFKC, quotes/dashes unified,
    '×' -> 'x', whitespace collapsed, case-folded. Words are NOT changed."""
    t = unicodedata.normalize("NFKC", text or "")
    t = t.translate(_DASHES).translate(_QUOTES).replace("×", "x")
    t = re.sub(r"-\s*\n\s*", "", t)  # re-join words hyphenated across lines
    t = re.sub(r"\s+", " ", t)
    return t.strip().casefold()


_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-/][a-z0-9]+)*")


def tokens(text: str) -> list[str]:
    """Keyword tokens. Keeps identifiers whole ('w10x49', '12.8.7', 'a992')
    and also emits their parts, so 'W10x49' matches both 'w10x49' and 'w10'."""
    out = []
    for tok in _TOKEN.findall(norm(text)):
        out.append(tok)
        parts = re.split(r"[.\-/x]", tok) if re.search(r"\d", tok) else []
        out.extend(p for p in parts if p and p != tok)
    return out


# ---------------------------------------------------------------- BM25 (no extra dependency)
class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.df: dict[str, int] = {}
        self.tf: list[dict[str, int]] = []
        self.len: list[int] = []
        self.avg = 0.0

    def fit(self, docs: list[str]) -> "BM25":
        for d in docs:
            c = Counter(tokens(d))
            self.tf.append(dict(c))
            self.len.append(sum(c.values()))
            for t in c:
                self.df[t] = self.df.get(t, 0) + 1
        self.avg = (sum(self.len) / len(self.len)) if self.len else 0.0
        return self

    def scores(self, query: str) -> list[float]:
        n = len(self.tf)
        q = tokens(query)
        out = [0.0] * n
        for t in set(q):
            df = self.df.get(t)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i, tf in enumerate(self.tf):
                f = tf.get(t)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * self.len[i] / (self.avg or 1))
                    out[i] += idf * f * (self.k1 + 1) / denom
        return out

    def to_json(self) -> dict:
        return {"k1": self.k1, "b": self.b, "df": self.df, "tf": self.tf, "len": self.len, "avg": self.avg}

    @classmethod
    def from_json(cls, d: dict) -> "BM25":
        m = cls(d["k1"], d["b"])
        m.df, m.tf, m.len, m.avg = d["df"], d["tf"], d["len"], d["avg"]
        return m


# ---------------------------------------------------------------- embeddings
class Embedder:
    """Two back-ends.
    st  : Sentence-Transformers model on your own machine (default; downloads weights once).
    lsa : offline fallback (hashed TF-IDF + SVD, numpy only). Weaker on paraphrases;
          use it only if you cannot download the model, and say so in your report."""

    def __init__(self, kind: str = "st", model: str = DEFAULT_ST_MODEL, state: dict | None = None):
        self.kind, self.model_name, self.state = kind, model, state or {}
        self._st = None

    # -- st
    def _load_st(self):
        if self._st is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                die("sentence-transformers is not installed. In your terminal (not inside Claude): "
                    "pip install -r .claude/skills/code-retriever/requirements.txt   "
                    "— or build with --embedder lsa (offline fallback).")
            self._st = SentenceTransformer(self.model_name)
        return self._st

    # -- lsa
    DIM_HASH, DIM_OUT = 4096, 256

    def _hash_tfidf(self, texts, idf=None):
        import numpy as np
        X = np.zeros((len(texts), self.DIM_HASH), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok, c in Counter(tokens(t)).items():
                X[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.DIM_HASH] += 1 + math.log(c)
        if idf is None:
            df = (X > 0).sum(0)
            idf = np.log((1 + len(texts)) / (1 + df)) + 1
        return X * idf, idf

    def fit_encode(self, texts: list[str]):
        import numpy as np
        if self.kind == "st":
            v = self._load_st().encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
            return np.asarray(v, dtype=np.float32)
        X, idf = self._hash_tfidf(texts)
        k = max(1, min(self.DIM_OUT, min(X.shape) - 1))
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        P = Vt[:k].T
        self.state = {"idf": idf.tolist(), "P": P.tolist()}
        return self._unit(X @ P)

    def encode(self, texts: list[str]):
        import numpy as np
        if self.kind == "st":
            prefix = "Represent this sentence for searching relevant passages: " if "bge" in self.model_name else ""
            v = self._load_st().encode([prefix + t for t in texts], normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(v, dtype=np.float32)
        idf = np.asarray(self.state["idf"], dtype=np.float32)
        P = np.asarray(self.state["P"], dtype=np.float32)
        X, _ = self._hash_tfidf(texts, idf)
        return self._unit(X @ P)

    @staticmethod
    def _unit(V):
        import numpy as np
        n = np.linalg.norm(V, axis=1, keepdims=True)
        n[n == 0] = 1
        return (V / n).astype(np.float32)


# ---------------------------------------------------------------- index I/O
def load_index():
    import numpy as np
    lock_p = INDEX / "manifest.lock"
    if not lock_p.exists():
        die("index/ not built yet — run: python3 .claude/skills/code-retriever/build_index.py")
    lock = json.loads(lock_p.read_text(encoding="utf-8"))
    chunks = [json.loads(l) for l in (INDEX / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    vectors = np.load(INDEX / "vectors.npy")
    bm25 = BM25.from_json(json.loads((INDEX / "bm25.json").read_text(encoding="utf-8")))
    state = {}
    if (INDEX / "embedder_state.json").exists():
        state = json.loads((INDEX / "embedder_state.json").read_text(encoding="utf-8"))
    emb = Embedder(lock["embedder"], lock.get("model", DEFAULT_ST_MODEL), state)
    return lock, chunks, vectors, bm25, emb


def stale_files(lock: dict) -> list[str]:
    """Files whose content changed (or appeared/disappeared) since the index was built."""
    out = []
    built = lock.get("files", {})
    for name, digest in built.items():
        p = CORPUS / name
        if not p.exists():
            out.append(f"{name} (missing)")
        elif sha256(p) != digest:
            out.append(f"{name} (changed)")
    for p in CORPUS.iterdir() if CORPUS.exists() else []:
        if p.suffix.lower() in SUPPORTED and p.name not in built:
            out.append(f"{p.name} (new, not indexed)")
    return out


# ---------------------------------------------------------------- search
def search(query: str, mode: str, k: int, chunks, vectors, bm25: BM25, emb: Embedder, rrf_k: int = 60):
    """Return [(chunk_index, score)] best first. hybrid = reciprocal rank fusion of both lists."""
    import numpy as np
    pool = max(k * 4, 20)
    ranked_e = ranked_k = []
    if mode in ("embedding", "hybrid"):
        sims = vectors @ emb.encode([query])[0]
        ranked_e = list(np.argsort(-sims)[:pool])
        if mode == "embedding":
            return [(int(i), float(sims[i])) for i in ranked_e[:k]]
    if mode in ("keyword", "hybrid"):
        s = bm25.scores(query)
        ranked_k = [i for i in sorted(range(len(s)), key=lambda i: -s[i])[:pool] if s[i] > 0]
        if mode == "keyword":
            return [(int(i), float(s[i])) for i in ranked_k[:k]]
    fused: dict[int, float] = {}
    for lst in (ranked_e, ranked_k):
        for r, i in enumerate(lst):
            fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (rrf_k + r + 1)
    return sorted(fused.items(), key=lambda x: -x[1])[:k]
