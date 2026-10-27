# Sensitivity of the Japanese text step on the check sentences

Conditions: checks/conditions/japanese_map_sensitivity.json (version 1, committed at b30607e before the run).
Result: results/japanese_map_sensitivity.json. 32 check sentences written by the makers; this describes the response
of the step, not its validity, and is not a result about any work.

- Coverage: in every combination the two sentences written to have no matched word (u1, u2) are the only unmatched
  sentences under the part-of-speech selection (2 of 32); under "all" none is unmatched. Every candidate occurrence
  was found in the lexicon (117 of 117 under the selection, 273 of 273 under "all").
- Clip: under maps A and B the clip changed no matched word occurrence and no sentence mean, so the two orders give
  the same points up to rounding (as expected for straight lines). Under map C the orders differ, through the
  nonlinearity of the rank.
- Main combination (B, word, part of speech), over the 30 matched sentences: V from -0.65 to 0.87, A from -0.25 to
  0.43; the rule's f from 4.51 to 5.86 (of its range 3 to 7). Map A gives a narrower spread (V -0.42 to 0.51, A -0.39
  to 0.01, nearly all A below 0); map C a much wider one (V and A close to -1 to 1).
- "All" (function words included) narrows the spread under A and B (main map: V -0.12 to 0.51, A -0.20 to 0.13).
- Groups (main combination, descriptive): the pleasant sentences have V 0.34 to 0.53, the unpleasant ones -0.65 to
  -0.05, the high-arousal ones A 0.08 to 0.33, the low-arousal ones A -0.15 to -0.11. The neutral sentences have A
  -0.25 to -0.11, i.e. as low as or lower than the low-arousal ones.
- Minimal pairs (main combination): negation is not handled. 好きではない gives the same point as 好きだ (the
  matched words are the same); 嬉しくない differs from 嬉しい only because the surface form 嬉しく has its own entry
  (V -0.24); 悲しくなかった differs from 悲しかった by V +0.03. The non-independent verbs of 泣いてしまった and the
  auxiliary of the polite form are left out, so those pairs differ only by the surface form of the main verb (0 to
  0.07).
- Under B the nominal neutral 5 of the lexicon maps to V 0.038 and A 0.415.
