# Reproducing the text example of the paper on the rule with the pinned generator

Conditions: `checks/conditions/reproduce_text_example.json` (version 3, committed at b223776 before the run). Script:
`checks/reproduce_text_example.py`. Result: `results/reproduce_text_example.json`.

- The generator installed by uv from `affective-relief-audit@b559a80` (commit in direct_url.json and SHA-256 of the
  five library files checked) reproduced every number stored for texts A and B (both variants, grids 128 and 1024):
  432 numbers, each with difference exactly 0 (tolerance 1e-12, judged one by one); the texts, the matched terms, Z,
  the unmatched flags and edge_argmax were equal and of the same type.
- Versions 1 and 2 of the check (same inputs) gave the same result; versions 2 and 3 were made after two rounds of
  independent reviews (see 'deviations' in the conditions). The structure of the reference is checked by name: a
  renamed text, a renamed variant and a dropped grid each stop the check.
- Sensitivity of the comparison (run interactively, not part of the conditions): with the seed 43 instead of 42 the
  first sentence of text B differed by up to 2.95 and one exact item (edge_argmax) differed; with eps changed by a
  relative 1e-9 the largest difference was 7.9e-9, above the tolerance. A NaN placed among other values, 3.0 for 3,
  1 for True, a shorter list, an infinite value and a missing key each gave a mismatch.
- Negative control (conditions, item 'negative_control'): the same pyproject.toml with the commit replaced by
  cec06910f0db5840c3801c18fd5e06893a1e16b6 was locked by uv in a temporary directory; its uv.lock records
  `rev=cec06910...#cec06910...`, while ours records `rev=b559a80c...#b559a80c...`. The library code of the two
  commits is the same, so this shows only that the commit is recorded.
