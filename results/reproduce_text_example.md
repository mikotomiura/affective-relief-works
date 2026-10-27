# Reproducing the text example of the paper on the rule with the pinned generator

Conditions: `checks/conditions/reproduce_text_example.json` (committed at a65e812 before the run). Script:
`checks/reproduce_text_example.py`. Result: `results/reproduce_text_example.json`.

- The generator installed by uv from `affective-relief-audit@b559a80` reproduced every number stored for texts A and B
  (both variants, grids 128 and 1024): 432 numbers, all with difference exactly 0 (tolerance 1e-12); the matched terms,
  Z, the unmatched flags and edge_argmax were equal.
- Sensitivity of the comparison (run interactively, not part of the conditions): with the seed 43 instead of 42 the
  first sentence of text B differed by up to 2.95 and one exact item (edge_argmax) differed; with eps changed by a
  relative 1e-9 the largest difference was 7.9e-9, above the tolerance.
- Negative control (conditions, item 'negative_control'): the same pyproject.toml with the commit replaced by
  cec06910f0db5840c3801c18fd5e06893a1e16b6 was locked by uv in a temporary directory; its uv.lock records
  `rev=cec06910...#cec06910...`, while ours records `rev=b559a80c...#b559a80c...`. The library code of the two
  commits is the same, so this shows only that the commit is recorded.
