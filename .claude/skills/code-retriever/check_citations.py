#!/usr/bin/env python3
"""Deterministic citation checker (Lab 5). No model inside.

    python3 .claude/skills/code-retriever/check_citations.py --json '<Grounded JSON>'
    python3 .claude/skills/code-retriever/check_citations.py answer.json

The answer must be a Grounded object:
    {
      "answer":    "L/360 for live load ...",
      "citations": [ { "chunk_id": "IBC2024_ch16:p016:02",
                       "doc": "IBC 2024 Chapter 16", "edition": "2024",
                       "section": "Table 1604.3", "page": 16,
                       "quote": "verbatim text from the chunk, at most 25 words" } ],
      "abstain":   false,
      "reason":    null
    }
If the corpus does not answer the question: "abstain": true, "citations": [], and a reason.

Verdicts (printed as JSON; exit code 0 = PASS or ABSTAIN, 1 = FAIL):
    PASS     every quote is in the chunk it cites, and doc / edition / section / page match it
    FAIL     malformed answer, unknown chunk, quote not verbatim, too long, or wrong doc/edition/section/page
    ABSTAIN  the answer declined, with a reason
PASS means grounded, not correct: you still compare the answer with your own reading.
Tip for the agent: inside the --json string write an apostrophe as \\u0027.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

from _common import INDEX, die, norm

MAX_QUOTE_WORDS = 25


def _flat(s: str) -> str:
    """norm() plus table separators removed — so a quoted table row matches with or without the pipes."""
    return re.sub(r"\s+", " ", norm(s).replace("|", " ")).strip()


def load_chunks_and_lock():
    lock_p = INDEX / "manifest.lock"
    if not lock_p.exists():
        die("index/ not built yet — run build_index.py first.")
    lock = json.loads(lock_p.read_text(encoding="utf-8"))
    by_id = {}
    for line in (INDEX / "chunks.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            c = json.loads(line)
            by_id[c["chunk_id"]] = c
    return by_id, lock


def validate_schema(obj) -> list[str]:
    errs = []
    if not isinstance(obj, dict):
        return ["answer is not a JSON object"]
    if not isinstance(obj.get("answer", ""), str):
        errs.append("'answer' must be a string")
    if not isinstance(obj.get("abstain", False), bool):
        errs.append("'abstain' must be true or false")
    cits = obj.get("citations", [])
    if not isinstance(cits, list):
        errs.append("'citations' must be a list")
        return errs
    for k, c in enumerate(cits):
        if not isinstance(c, dict):
            errs.append(f"citation {k}: not an object")
            continue
        for f in ("chunk_id", "doc", "edition", "section", "page", "quote"):
            if c.get(f) in (None, ""):
                errs.append(f"citation {k}: missing '{f}'")
    if not obj.get("abstain", False) and not cits:
        errs.append("no citations and not an abstain — every answer needs at least one citation")
    if obj.get("abstain", False) and not str(obj.get("reason") or "").strip():
        errs.append("abstain without a reason")
    return errs


def check(obj, by_id: dict, lock: dict) -> dict:
    errs = validate_schema(obj)
    if errs:
        return {"verdict": "FAIL", "errors": ["schema: " + e for e in errs], "checks": []}
    if obj.get("abstain"):
        return {"verdict": "ABSTAIN", "errors": [], "checks": [], "reason": obj.get("reason")}
    checks, errors = [], []
    for k, c in enumerate(obj["citations"]):
        res = {"citation": k, "chunk_id": c["chunk_id"], "problems": []}
        ch = by_id.get(c["chunk_id"])
        if ch is None:
            res["problems"].append("unknown chunk_id (not in the index)")
        else:
            q = str(c["quote"])
            if len(q.split()) > MAX_QUOTE_WORDS:
                res["problems"].append(f"quote has {len(q.split())} words (max {MAX_QUOTE_WORDS})")
            if not (norm(q) in norm(ch["text"]) or _flat(q) in _flat(ch["text"])):
                res["problems"].append("quote is not verbatim in the cited chunk")
            pinned = lock.get("documents", {}).get(ch["file"], {}).get("edition", ch["edition"])
            if norm(str(c["edition"])) != norm(str(pinned)):
                res["problems"].append(f"edition '{c['edition']}' does not match the pinned edition '{pinned}'")
            if norm(str(c["doc"])) != norm(ch["doc"]):
                res["problems"].append(f"doc '{c['doc']}' does not match the chunk's document '{ch['doc']}'")
            # the cited section may be shorter or longer than the chunk's heading, but must overlap it
            sec_c = norm(str(c["section"]))
            own = [x for x in (norm(ch["section"] or ""), norm(ch.get("table") or "")) if x]
            if not any(sec_c in x or x in sec_c for x in own):
                res["problems"].append(f"section '{c['section']}' does not match the chunk's section '{ch['section']}'")
            try:
                p = int(str(c["page"]).strip().lstrip("p.").strip())
                if not (ch["page"] <= p <= ch.get("page_end", ch["page"])):
                    res["problems"].append(f"page {p} is not the chunk's page ({ch['page']}–{ch.get('page_end', ch['page'])})")
            except ValueError:
                res["problems"].append(f"page '{c['page']}' is not a page number")
        res["ok"] = not res["problems"]
        checks.append(res)
        errors += [f"citation {k}: {p}" for p in res["problems"]]
    return {"verdict": "PASS" if not errors else "FAIL", "errors": errors, "checks": checks}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", help="a file holding the Grounded JSON (or use --json, or stdin)")
    ap.add_argument("--json", dest="json_text", help="the Grounded JSON as one argument")
    a = ap.parse_args()
    raw = a.json_text if a.json_text is not None else (open(a.file, encoding="utf-8").read() if a.file else sys.stdin.read())
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        print(json.dumps({"verdict": "FAIL", "errors": [f"schema: not valid JSON ({e})"], "checks": []}, indent=1))
        sys.exit(1)
    by_id, lock = load_chunks_and_lock()
    out = check(obj, by_id, lock)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    sys.exit(0 if out["verdict"] in ("PASS", "ABSTAIN") else 1)


if __name__ == "__main__":
    main()
