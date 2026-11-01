# LOG

## 2026-11-01 · Lab 5, Checkpoint B — did the table survive chunking?

Clause checked: Tien et al. 2013 (PLOS ONE), Table 1, maximum allowed ASA per residue (Å²).

**Default text parser** (`build_index.py`, 713 chunks, 0 table chunks): the table was shredded.
The caption and column headers ended chunk `tien_sasa:p003:02`; the rows ran on into
`tien_sasa:p003:03` with no headers, and the Alanine row was split across the boundary.
Section labels were fragments ("0. We then rotated the q and").
The query "theoretical maximum solvent accessible surface area of alanine" returned only
prose about the values, never the table.

**Fix: rebuilt with the layout parser** (`build_index.py --parser layout`, 450 chunks, 22 table chunks).
The same query now returns the table as one chunk, with headers on every row:

```
tien_sasa:p003:01 · Tien et al., Maximum allowed solvent accessibilities of residues in proteins (2013) · p. 3 · Table on p. 3
Residue | Theoretical | Empirical | Milleretal.(1987) | Roseetal.(1985)
Alanine | 129.0 | 121.0 | 113.0 | 118.1
Arginine | 274.0 | 265.0 | 241.0 | 256.0
Asparagine | 195.0 | 187.0 | 158.0 | 165.5
...
```

Structure survived: rows intact, headers present, page correct. Two remaining defects:
the chunk's section label is wrong ("Figure 2. Ramachandran plot for alanine residues in our",
inherited from p. 2), and the layout parser runs two-column prose together and drops spaces
(e.g. "MaximumSolventAccessibilities"), which may hurt keyword search on prose. Worth comparing
hit@5 for both parsers on the same Part D questions.

`check_citations.py` on this chunk: PASS for the verbatim Alanine row; FAIL when 129.0 is changed to 130.0.

TODO (Ian): check the values above against the printed table before submitting.
