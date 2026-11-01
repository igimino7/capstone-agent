# code-retriever — skill #3 of the capstone agent (ME/CE/AM 295, Lab 5)

A retriever over your own project's documents, and the scripts that measure it.
The Lab 5 handout is the walkthrough; this page is the reference.

## Files

| File | Who runs it | What it does |
|---|---|---|
| `SKILL.md` | the agent reads it | When and how to retrieve, answer under the Grounded schema, and check |
| `retrieve.py` | agent and you | Top-5 chunks as JSON; `--show` prints them for a human with their chunk ids |
| `check_citations.py` | agent and you | Deterministic PASS / FAIL / ABSTAIN on a Grounded answer. No model inside |
| `build_index.py` | you, in the terminal | `--check` the manifest; build `index/` (`--parser text` or `layout`) |
| `evaluate.py` | you, in the terminal | hit@5 per mode (D.2); optional 20-run tally (D.3) |
| `_common.py` | — | Shared code: paths, normalisation, BM25, embeddings, hybrid search |
| `examples/` | you | A sample `manifest.json` and `questions.jsonl` to copy and edit |

## Setup (once, in your terminal)

```bash
pip install -r .claude/skills/code-retriever/requirements.txt
mkdir -p corpus eval
cp .claude/skills/code-retriever/examples/manifest.example.json corpus/manifest.json   # then edit it
```

Add to `.claude/settings.json` (the agent may run only the two retrieval scripts):

```json
"allow": [ "Bash(python3 .claude/skills/code-retriever/retrieve.py *)",
           "Bash(python3 .claude/skills/code-retriever/check_citations.py *)" ],
"deny":  [ "Edit(corpus/**)", "Edit(index/**)", "WebFetch", "WebSearch" ]
```

## Commands

```bash
python3 .claude/skills/code-retriever/build_index.py --check            # Part A
python3 .claude/skills/code-retriever/build_index.py                    # Part B (text-stream parser)
python3 .claude/skills/code-retriever/build_index.py --parser layout    # if tables come out shredded
python3 .claude/skills/code-retriever/retrieve.py --show "web thickness W10x49"
python3 .claude/skills/code-retriever/check_citations.py --json '{"answer": "...", ...}'
python3 .claude/skills/code-retriever/evaluate.py --mode embedding      # Part D.2
python3 .claude/skills/code-retriever/evaluate.py --mode hybrid
python3 .claude/skills/code-retriever/evaluate.py --mode keyword        # optional third row
python3 .claude/skills/code-retriever/evaluate.py --mode hybrid --repeat 20 --questions q03,q07,q12,q18,q22   # D.3, optional
```

## What is in `index/`

`chunks.jsonl` (text + metadata: doc, edition, section, page, table, chunk_id), `vectors.npy`
(embeddings), `bm25.json` (keyword index), `manifest.lock` (parser, embedding model, chunk size,
and a SHA-256 of every corpus file — the scripts warn when the index is older than the corpus).

## Design notes

- **Hybrid** = embedding search and BM25 keyword search, merged by reciprocal rank fusion (k = 60).
- **Keyword tokens keep identifiers whole and split them**: `W10x49` matches `w10x49`, `w10` and `49`;
  `×` is read as `x`.
- **Layout parser**: pdfplumber table detection; each row is kept with its column headers, and a blank
  or abbreviated first cell inherits the group label from the row above (`×49` under `W10×112` → `W10x49`).
  Footnote lines under a table on the same page are attached to the table's chunks. Check the result with
  `--show` — no parser is right for every PDF.
- **Embeddings**: `BAAI/bge-small-en-v1.5` via Sentence-Transformers, on your machine; no key, nothing leaves
  the computer after the one-time download. `--embedder lsa` is an offline fallback (numpy only) for machines
  that cannot download the model; it is weaker on paraphrases — say so in your report if you use it.
- **The check is deterministic**: quote ⊂ chunk text (after normalising whitespace, quotes, dashes, `×`),
  ≤ 25 words; doc, edition (against the pin in `manifest.json`), section and page must match the chunk.
  PASS means grounded, not correct.
- **The full pipeline in `evaluate.py`** answers with `claude -p` (your pinned model, from this repo's
  settings) and runs the check itself, so the numbers do not depend on the model choosing to call it.
  Each run is saved in `eval/runs/`.
- Scanned PDFs (images only) produce no text; they need OCR before they can be indexed.
