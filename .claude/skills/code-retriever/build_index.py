#!/usr/bin/env python3
"""Build the retrieval index for the code-retriever skill (Lab 5, Parts A–B).

    python3 .claude/skills/code-retriever/build_index.py --check
        Check corpus/manifest.json against corpus/: page counts, missing fields,
        files not in the manifest and manifest entries without a file. Builds nothing.

    python3 .claude/skills/code-retriever/build_index.py [--parser text|layout] [--embedder st|lsa]
        Parse -> chunk -> embed -> index. Writes index/chunks.jsonl, index/vectors.npy,
        index/bm25.json, index/manifest.lock.

--parser text    (default) the PDF's text stream, page by page. Fast. Tables come out
                 as runs of numbers: the W10x49 problem from Lecture 8.
--parser layout  pdfplumber table detection: each table row is kept whole, with its
                 column headers, and a blank or abbreviated first cell (e.g. "x49")
                 inherits the group label from the row above ("W10x49").
--embedder st    (default) Sentence-Transformers model on your machine, no API key.
--embedder lsa   offline fallback (numpy only) if you cannot download the model.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys

from _common import (CORPUS, DEFAULT_ST_MODEL, INDEX, REQUIRED_FIELDS, ROOT, SUPPORTED,
                     BM25, Embedder, die, load_manifest, sha256)

HEADING = re.compile(
    r"^(?:"
    r"(?:CHAPTER|Chapter|SECTION|Section|APPENDIX|Appendix|TABLE|Table|FIGURE|Figure)\s+[A-Z0-9][\w.\-]*\b.{0,90}"
    r"|§\s*\d+(?:\.\d+)*.{0,90}"
    r"|\d+(?:\.\d+){0,4}\.?\s+[A-Z][^.]{2,90}"
    r")$"
)
FOOTNOTE = re.compile(r"^(?:[a-h†‡*]\s|\([a-h]\)\s|Notes?[:.]\s|NOTES?[:.]\s|Source:)")


# ---------------------------------------------------------------- page readers
def page_count(path) -> int:
    if path.suffix.lower() != ".pdf":
        return 1
    from pypdf import PdfReader
    return len(PdfReader(str(path)).pages)


def read_text_pages(path) -> list[dict]:
    """[{page, lines, tables}] using the plain text stream."""
    if path.suffix.lower() != ".pdf":
        return [{"page": 1, "lines": path.read_text(encoding="utf-8", errors="replace").splitlines(), "tables": []}]
    from pypdf import PdfReader
    out = []
    for i, pg in enumerate(PdfReader(str(path)).pages, 1):
        out.append({"page": i, "lines": (pg.extract_text() or "").splitlines(), "tables": []})
    return out


def _fill_group_labels(rows: list[list[str]]) -> list[list[str]]:
    """Carry the group label down the first column: '' -> previous label,
    'x49' / '×49' -> 'W10x49' when the previous label was 'W10x112'."""
    last = ""
    out = []
    for r in rows:
        r = [(c or "").replace("\n", " ").strip() for c in r]
        if r:
            c0 = r[0]
            if not c0 and last:
                r[0] = last
            elif re.match(r"^[x×]\s*\d", c0) and re.match(r"^[A-Za-z]+\d+[x×]", last):
                r[0] = re.match(r"^([A-Za-z]+\d+)[x×]", last).group(1) + "x" + re.sub(r"^[x×]\s*", "", c0)
            last = r[0] or last
        out.append(r)
    return out


def read_layout_pages(path) -> list[dict]:
    """[{page, lines, tables}] with tables extracted as whole rows (pdfplumber)."""
    if path.suffix.lower() != ".pdf":
        return read_text_pages(path)
    import pdfplumber
    out = []
    with pdfplumber.open(str(path)) as pdf:
        for i, pg in enumerate(pdf.pages, 1):
            tables, boxes = [], []
            for t in pg.find_tables():
                rows = [r for r in t.extract() if r and any((c or "").strip() for c in r)]
                if len(rows) >= 2:
                    tables.append(_fill_group_labels(rows))
                    boxes.append(t.bbox)

            def outside(obj, boxes=boxes):
                x, y = (obj.get("x0", 0) + obj.get("x1", 0)) / 2, (obj.get("top", 0) + obj.get("bottom", 0)) / 2
                return not any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in boxes)

            text = pg.filter(outside).extract_text() or ""
            out.append({"page": i, "lines": text.splitlines(), "tables": tables})
    return out


# ---------------------------------------------------------------- chunking
def split_words(text: str, max_words: int, overlap: int) -> list[str]:
    """Split a section into pieces of <= max_words, cutting at sentence ends where possible."""
    words = text.split()
    if len(words) <= max_words:
        return [text.strip()] if words else []
    pieces, start = [], 0
    while start < len(words):
        end = min(len(words), start + max_words)
        if end < len(words):
            for j in range(end, start + max_words // 2, -1):
                if re.search(r"[.;:]$", words[j - 1]):
                    end = j
                    break
        pieces.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = max(end - overlap, start + 1)
    return pieces


def chunk_document(entry: dict, pages: list[dict], max_words: int, overlap: int) -> list[dict]:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", entry["file"].rsplit(".", 1)[0])
    base = {"doc": entry["name"], "edition": str(entry["edition"]), "file": entry["file"]}
    chunks, per_page = [], {}

    def new_id(page):
        per_page[page] = per_page.get(page, 0) + 1
        return f"{stem}:p{page:03d}:{per_page[page]:02d}"

    section, buf, buf_page = "(start of document)", [], 1

    def flush(end_page):
        text = " ".join(buf).strip()
        for piece in split_words(text, max_words, overlap):
            chunks.append({**base, "chunk_id": new_id(buf_page), "section": section,
                           "page": buf_page, "page_end": end_page, "table": None, "text": piece})

    last_table_chunks: list[dict] = []
    for pg in pages:
        p = pg["page"]
        # tables first (layout parser only): one chunk per ~25 rows, header repeated
        last_table_chunks = []
        title = next((l.strip() for l in pg["lines"] if re.match(r"^\s*(TABLE|Table)\s+\S+", l)), f"Table on p. {p}")
        for t in pg["tables"]:
            header, body = t[0], t[1:]
            for s in range(0, len(body), 25):
                rows = body[s:s + 25]
                text = title + "\n" + " | ".join(header) + "\n" + "\n".join(" | ".join(r) for r in rows)
                c = {**base, "chunk_id": new_id(p), "section": title if section.startswith("(") else section,
                     "page": p, "page_end": p,
                     "table": title, "text": text}
                chunks.append(c)
                last_table_chunks.append(c)
        for line in pg["lines"]:
            s_line = line.strip()
            if not s_line:
                continue
            if last_table_chunks and FOOTNOTE.match(s_line):
                for c in last_table_chunks:  # keep the footnote with the table it governs
                    c["text"] += "\n" + s_line
            if HEADING.match(s_line) and len(s_line) <= 100 and not s_line.endswith((".", ",", ";")):
                if buf:
                    flush(p)
                section, buf, buf_page = s_line, [], p
                continue
            if not buf:
                buf_page = p
            buf.append(s_line)
    if buf:
        flush(pages[-1]["page"] if pages else 1)
    return chunks


# ---------------------------------------------------------------- check
def run_check(entries: list[dict]) -> int:
    problems, rows = [], []
    listed = set()
    for i, e in enumerate(entries, 1):
        missing = [f for f in REQUIRED_FIELDS if not str(e.get(f) or "").strip()]
        f = e.get("file")
        if missing:
            problems.append(f"entry {i} ({e.get('name', '?')}): missing {', '.join(missing)}")
        if not f:
            rows.append((e.get("name", "?"), str(e.get("edition", "?")), e.get("licence", "?"), "-", "no file (not indexed)"))
            continue
        listed.add(f)
        p = CORPUS / f
        if " " in f:
            problems.append(f"{f}: filename has spaces — rename it (and the manifest entry)")
        if not p.exists():
            problems.append(f"{f}: listed in manifest.json but not in corpus/")
            rows.append((e.get("name", "?"), str(e.get("edition", "?")), e.get("licence", "?"), "-", "MISSING"))
            continue
        if p.suffix.lower() not in SUPPORTED:
            problems.append(f"{f}: unsupported type (use PDF, .txt or .md)")
        try:
            n = page_count(p)
        except Exception as ex:  # unreadable or encrypted PDF
            n, _ = "?", problems.append(f"{f}: cannot read ({ex.__class__.__name__}: {ex})")
        rows.append((e.get("name", "?"), str(e.get("edition", "?")), e.get("licence", "?"), str(n), "ok"))
    for p in sorted(CORPUS.iterdir()):
        if p.name == "manifest.json" or p.name.startswith("."):
            continue
        if p.name not in listed:
            problems.append(f"{p.name}: in corpus/ but not in manifest.json")
    w = [max(len(r[k]) for r in rows + [("document", "edition", "licence", "pages", "status")]) for k in range(5)]
    for r in [("document", "edition", "licence", "pages", "status")] + rows:
        print("  ".join(c.ljust(w[k]) for k, c in enumerate(r)))
    print()
    if problems:
        print(f"{len(problems)} problem(s):")
        for x in problems:
            print("  - " + x)
        return 1
    print(f"OK: {len(rows)} documents, manifest and corpus/ agree.")
    return 0


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="check the manifest against corpus/ and stop")
    ap.add_argument("--parser", choices=["text", "layout"], default="text")
    ap.add_argument("--embedder", choices=["st", "lsa"], default="st")
    ap.add_argument("--model", default=DEFAULT_ST_MODEL, help="Sentence-Transformers model (with --embedder st)")
    ap.add_argument("--chunk-words", type=int, default=350, help="max words per text chunk (~ 450 tokens)")
    ap.add_argument("--overlap", type=int, default=50, help="words repeated between consecutive chunks")
    a = ap.parse_args()

    entries = load_manifest()
    if a.check:
        sys.exit(run_check(entries))
    if run_check(entries) != 0:
        die("fix the manifest problems above, then build again.", 1)

    reader = read_layout_pages if a.parser == "layout" else read_text_pages
    chunks, files = [], {}
    for e in entries:
        if not e.get("file"):
            continue
        p = CORPUS / e["file"]
        pages = reader(p)
        c = chunk_document(e, pages, a.chunk_words, a.overlap)
        if not c:
            print(f"warning: {e['file']} produced no text — scanned PDF? It needs OCR first.", file=sys.stderr)
        chunks += c
        files[e["file"]] = sha256(p)
    if not chunks:
        die("no chunks produced — nothing to index.", 1)

    import numpy as np
    emb = Embedder(a.embedder, a.model)
    texts = [f"{c['doc']} | {c['section']} | {c['table'] or ''}\n{c['text']}" for c in chunks]
    vectors = emb.fit_encode(texts)
    bm25 = BM25().fit(texts)

    INDEX.mkdir(exist_ok=True)
    with open(INDEX / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    np.save(INDEX / "vectors.npy", vectors)
    (INDEX / "bm25.json").write_text(json.dumps(bm25.to_json()), encoding="utf-8")
    if emb.state:
        (INDEX / "embedder_state.json").write_text(json.dumps(emb.state), encoding="utf-8")
    elif (INDEX / "embedder_state.json").exists():
        (INDEX / "embedder_state.json").unlink()
    lock = {
        "built": dt.datetime.now().isoformat(timespec="seconds"),
        "parser": a.parser, "embedder": a.embedder,
        "model": a.model if a.embedder == "st" else "lsa-hash4096-svd256",
        "chunk_words": a.chunk_words, "overlap": a.overlap, "n_chunks": len(chunks),
        "files": files,
        "documents": {e["file"]: {"name": e["name"], "edition": str(e["edition"])} for e in entries if e.get("file")},
    }
    (INDEX / "manifest.lock").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    n_tables = sum(1 for c in chunks if c["table"])
    print(f"-> parsed {len(files)} documents, {len(chunks):,} chunks ({n_tables} table chunks), parser={a.parser}")
    print(f"-> index/chunks.jsonl, index/vectors.npy, index/bm25.json, index/manifest.lock  (in {ROOT.name}/)")


if __name__ == "__main__":
    main()
