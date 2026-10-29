"""The template for noting decisions: the quantities a rule leaves open and the decisions changed after the values were
seen, with the reason, the alternatives tried, the images that used each one and who judged the look, filled in for works 1
and 2, and the same table empty as its own image (for supplement S1, version 11; lines of note under it, version 12, its own
general lines, version 13), from the conditions and the files they name.

Usage:  uv run python works/decision_note.py        (refuses to overwrite)
Conditions: works/conditions/decision_note.json, version 17 (committed before this version of the script was written)
Input:  the files named in the rows (conditions of works 1 and 2, of the drawing, of the blind judgement and of the records
        of the share used, the scoring of the blind judgement and its supplement, and the images each quantity acted on) and
        the values every share is computed again from (results/work1/work1.json, results/work2_words/words.json,
        results/paper_words_values/words_values.json, and the shuffled sentences of results/paper_work1_shuffled
        shuffled again), and the records of the sets beside map B and the percentiles (results/paper_reference_range,
        version 13, and results/paper_distinct_range, version 14, read and checked by their provenance); every one
        must be the committed bytes at HEAD. The range of f and s
        is read from the installed atact.mapping, which must be the library of every record of the share used
Output: results/decision_note_v17/decision_note.png (figure 8), decision_note_empty.png (S1), decision_note.md and
        decision_note.json
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.pycache_prefix = tempfile.mkdtemp(prefix="pycache-")  # never run a stale .pyc (ignored by git)
sys.path.insert(0, str(ROOT / "works"))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import _common as cm  # noqa: E402

COND = ROOT / "works" / "conditions" / "decision_note.json"
OUT_DIR = ROOT / "results" / "decision_note_v17"
CODE = ["works/decision_note.py", "works/_common.py"]
MM_PER_INCH = 25.4
VERSION = 17
ROW_SETS = {"rows_work1": "Work 1", "rows_work2": "Work 2"}
# the ids of the rows, in order (checks of the conditions); the number of rows follows from them
ROW_IDS = {"rows_work1": ["w1_lexicon", "w1_height_scale", "w1_drawing"], "rows_work2": ["w2_map_b", "w2_percentiles", "w2_seed"]}
COLUMNS = ["quantity", "value", "reason", "acted_on", "when", "judged"]
ROW_FIELDS = {"id", "quantity", "value", "reason", "acted_on", "compared", "when", "judged", "dagger"}
NONE_RECORDED = "none recorded"
# answers_rule of the conditions: the phrase of each outcome and of each row is fixed here, not in the rows
OUTCOME_PHRASE = {"both differ": "both said the rows differ", "neither": "neither said the rows differ", "split": "they split"}
ROW_PHRASE = {"words20": "the 20 words"}
# the version each row of work 2 speaks of (G0: map B, P: the percentiles, SPC: the seed with the color, of
# blind_judgement.json; the seed from version 13)
VERSION_OF = {"w2_map_b": "G0", "w2_percentiles": "P", "w2_seed": "SPC"}
# approved_text_rule of the conditions: every cell of every row as written in the conditions (before a phrase is built),
# with its kind, character for character
APPROVED_CELLS = {
    "w1_lexicon": {
        "quantity": "lexicon; sentence as mean",
        "value": ("text", "NRC VAD v2.1"),
        "reason": "Human ratings of English words. Tried: none. Later compared with: shuffled and drawn sentences.",
        "when": "before; seen: none",
        "judged": ("none", NONE_RECORDED)},
    "w1_height_scale": {
        "quantity": "height scale",
        "value": ("format", "common to the series, assumed relief depth D = {D_mm} mm"),
        "reason": "Keeps the depth differences between sentences. Tried: none. Shown beside: per-tile min-max.",
        "when": "before; seen: none",
        "judged": ("none", NONE_RECORDED)},
    "w1_drawing": {
        "quantity": "drawing",
        "value": ("literal", "four shown together, the series using the gray of the height; gray; lit at 45 degrees from "
                             "upper left, lower right; stretched per tile"),
        "reason": "Also a quantity the maker sets (one viewpoint). Tried: none. Shown beside: the other three. "
                  "Not tried: other viewpoints.",
        "when": "before; seen: earlier images and values of work 1",
        "judged": ("none", NONE_RECORDED)},
    "w2_map_b": {
        "quantity": "lexicon and map",
        "value": ("text", "MEmoLon (Japanese) through map B"),
        "reason": "Brings the Japanese values onto the English scale. Tried: none. Later compared with: those beside; "
                  "the percentiles; two other maps, on check sentences.",
        "when": "before; seen: both lexicons' values",
        "judged": ("text", "two makers (one blind), with the sheet and example")},
    "w2_percentiles": {
        "quantity": "percentiles",
        "value": ("text", "V and A as percentiles in the 23 works, each author weighted equally"),
        "reason": "Spreads values map B compresses; stretches small predicted differences, most in A. Tried: map B; "
                  "whole-lexicon percentiles.",
        "when": "after; seen: sketches with these percentiles, whole-lexicon percentiles",
        "judged": ("text", "same makers and sheet")},
    "w2_seed": {
        "quantity": "seed",
        "value": ("literal", "per word: SHA-256 of the word plus a common seed"),
        "reason": "Varies the pattern by word; rows can differ by spelling alone. Tried: per-author seeds; a common seed.",
        "when": "after; seen: sketches with this per-word rule",
        "judged": ("text", "judged only in the version with color")},
}
APPROVED_TEXT = {(rid, f): cells[f][1] for rid, cells in APPROVED_CELLS.items() for f in ("value", "judged")
                 if cells[f][0] == "text"}
# approved_text_rule of the conditions (version 14): every row as written in the conditions, and the frame of the table
APPROVED_ROWS = {'w1_lexicon': {'id': 'w1_lexicon',
                'quantity': 'lexicon; sentence as mean',
                'value': {'text': 'NRC VAD v2.1',
                          'check': {'file': 'works/conditions/work1.json',
                                    'keys': ['lexicon.name', 'steps'],
                                    'must_contain': ['NRC VAD Lexicon v2.1', 'every matched term averaged']},
                          'used_range': {'file': 'results/paper_work1_same_unit/same_unit.json',
                                         'set': 'sentences',
                                         'n': 104,
                                         'unit': 'sentences',
                                         'shown': {'f': '7%', 's': '6%'},
                                         'beside': [{'file': 'results/paper_work1_shuffled/shuffled.json',
                                                     'set': 'shuffled_sentences',
                                                     'label': 'terms shuffled among them',
                                                     'shown': {'f': '6%', 's': '6%'}},
                                                    {'file': 'results/paper_work1_same_unit/same_unit.json',
                                                     'set': 'random_sentences',
                                                     'label': 'uniform draws, same term counts',
                                                     'shown': {'f': '10%', 's': '10%'}}]}},
                'reason': {'why': 'human ratings of English words',
                           'tried': [],
                           'shown_beside': [],
                           'later_compared': ['shuffled and drawn sentences'],
                           'not_tried': []},
                'acted_on': ['results/work1/common.png',
                             'results/work1/minmax.png',
                             'results/work1/common_eps.png',
                             'results/work1_drawing/drawing.png'],
                'compared': [],
                'when': 'before; seen: none',
                'judged': 'none recorded',
                'dagger': False},
 'w1_height_scale': {'id': 'w1_height_scale',
                     'quantity': 'height scale',
                     'value': {'file': 'works/conditions/work1.json',
                               'keys': ['relief.D_mm'],
                               'format': 'common to the series, assumed relief depth D = {D_mm} mm'},
                     'reason': {'why': 'keeps the depth differences between sentences',
                                'tried': [],
                                'shown_beside': ['per-tile min-max'],
                                'later_compared': [],
                                'not_tried': []},
                     'acted_on': ['results/work1/common.png',
                                  'results/work1/minmax.png',
                                  'results/work1/common_eps.png',
                                  'results/work1_drawing/drawing.png'],
                     'compared': ['results/work1/common.png', 'results/work1/minmax.png'],
                     'when': 'before; seen: none',
                     'judged': 'none recorded',
                     'dagger': False},
 'w1_drawing': {'id': 'w1_drawing',
                'quantity': 'drawing',
                'value': {'literal': 'four shown together, the series using the gray of the height; gray; lit at 45 '
                                     'degrees from upper left, lower right; stretched per tile',
                          'check': {'file': 'works/conditions/work1_drawing.json',
                                    'keys': ['drawings.a_height',
                                             'drawings.b_light_upper_left',
                                             'drawings.c_light_lower_right'],
                                    'must_contain': ['z / D + 1/2',
                                                     'elevation 45 degrees',
                                                     'azimuth 135 degrees',
                                                     'azimuth 315 degrees']}},
                'reason': {'why': 'also a quantity the maker sets (one viewpoint)',
                           'tried': [],
                           'shown_beside': ['the other three'],
                           'later_compared': [],
                           'not_tried': ['other viewpoints']},
                'acted_on': ['results/work1_drawing/drawing.png'],
                'compared': ['results/work1_drawing/drawing.png'],
                'when': 'before; seen: earlier images and values of work 1',
                'judged': 'none recorded',
                'dagger': False},
 'w2_map_b': {'id': 'w2_map_b',
              'quantity': 'lexicon and map',
              'value': {'text': 'MEmoLon (Japanese) through map B',
                        'check': {'file': 'works/conditions/work2.json',
                                  'keys': ['lexicon.name', 'map.use'],
                                  'must_contain': ['MEmoLon', 'Japanese', 'map B']},
                        'used_range': {'file': 'results/paper_used_range/used_range.json',
                                       'set': 'main_G0.pooled',
                                       'n': 120,
                                       'unit': 'words',
                                       'shown': {'f': '8%', 's': '12%'},
                                       'beside': [{'file': 'results/paper_reference_range/reference_range.json',
                                                   'set': 'lexicon_B',
                                                   'label': 'whole lexicon',
                                                   'shown': {'f': '7%', 's': '11%'}},
                                                  {'file': 'results/paper_distinct_range/distinct_range.json',
                                                   'set': 'distinct_B',
                                                   'label': "23 works' words",
                                                   'shown': {'f': '12%', 's': '21%'}}]}},
              'reason': {'why': 'brings the Japanese values onto the English scale',
                         'tried': [],
                         'shown_beside': [],
                         'later_compared': ['those beside', 'the percentiles', 'two other maps, on check sentences'],
                         'not_tried': []},
              'acted_on': ['results/paper_figure6_scale/figure6_scale.png',
                           'results/paper_work2_figures/variants_judged.png'],
              'compared': ['results/paper_figure6_scale/figure6_scale.png'],
              'when': "before; seen: both lexicons' values",
              'judged': {'text': 'two makers (one blind), with the sheet and example',
                         'check': {'file': 'works/conditions/blind_judgement.json',
                                   'keys': ['judges.deciding', 'judges.also'],
                                   'must_contain': ['has not seen the sheets', 'not as a blind judgement']},
                         'answers': {'main': {'file': 'results/blind_judgement/score.json',
                                              'key': 'decision.rows_remaining',
                                              'decided_key': 'decision.decided'},
                                     'also': {'file': 'results/blind_judgement/score-researcher-supplement.json',
                                              'key': 'rows_remaining',
                                              'pass_key': 'controls_pass'},
                                     'row': 'words20',
                                     'version': 'G0',
                                     'outcome': 'split'}},
              'dagger': False},
 'w2_percentiles': {'id': 'w2_percentiles',
                    'quantity': 'percentiles',
                    'value': {'text': 'V and A as percentiles in the 23 works, each author weighted equally',
                              'check': {'file': 'works/conditions/sketch_sp_color.json',
                                        'keys': ['reference.what'],
                                        'must_contain': ['the 23 works', 'each author weighs 1']},
                              'used_range': {'file': 'results/paper_used_range/used_range.json',
                                             'set': 'main_P.pooled',
                                             'n': 120,
                                             'unit': 'words',
                                             'shown': {'f': '54%', 's': '60%'},
                                             'beside': [{'file': 'results/paper_distinct_range/distinct_range.json',
                                                         'set': 'distinct_P',
                                                         'label': "23 works' words",
                                                         'shown': {'f': '49%', 's': '36%'}}]}},
                    'reason': {'why': 'spreads values map B compresses; stretches small predicted differences, most in '
                                      'A',
                               'tried': ['map B', 'whole-lexicon percentiles'],
                               'shown_beside': [],
                               'later_compared': [],
                               'not_tried': []},
                    'acted_on': ['results/paper_figure6_scale/figure6_scale.png', 'results/work2_sp_color/SPC.png'],
                    'compared': ['results/paper_figure6_scale/figure6_scale.png'],
                    'when': 'after; seen: sketches with these percentiles, whole-lexicon percentiles',
                    'judged': {'text': 'same makers and sheet',
                               'check': {'file': 'works/conditions/blind_judgement.json',
                                         'keys': ['judges.deciding', 'judges.also'],
                                         'must_contain': ['has not seen the sheets', 'not as a blind judgement']},
                               'answers': {'main': {'file': 'results/blind_judgement/score.json',
                                                    'key': 'decision.rows_remaining',
                                                    'decided_key': 'decision.decided'},
                                           'also': {'file': 'results/blind_judgement/score-researcher-supplement.json',
                                                    'key': 'rows_remaining',
                                                    'pass_key': 'controls_pass'},
                                           'row': 'words20',
                                           'version': 'P',
                                           'outcome': 'both differ'}},
                    'dagger': False},
 'w2_seed': {'id': 'w2_seed',
             'quantity': 'seed',
             'value': {'literal': 'per word: SHA-256 of the word plus a common seed',
                       'check': {'file': 'works/conditions/work2_sp_color.json',
                                 'keys': ['seed'],
                                 'must_contain': ['per word', 'SHA-256 of the UTF-8 word', '+ the common seed']}},
             'reason': {'why': 'varies the pattern by word; rows can differ by spelling alone',
                        'tried': ['per-author seeds', 'a common seed'],
                        'shown_beside': [],
                        'later_compared': [],
                        'not_tried': []},
             'acted_on': ['results/work2_sp_color/SPC.png'],
             'compared': [],
             'when': 'after; seen: sketches with this per-word rule',
             'judged': {'text': 'judged only in the version with color',
                        'check': {'file': 'works/conditions/blind_judgement.json',
                                  'keys': ['judges.deciding', 'judges.also', 'versions.SPC'],
                                  'must_contain': ['has not seen the sheets',
                                                   'not as a blind judgement',
                                                   'the seed per spelling',
                                                   'coloured']},
                        'answers': {'main': {'file': 'results/blind_judgement/score.json',
                                             'key': 'decision.rows_remaining',
                                             'decided_key': 'decision.decided'},
                                    'also': {'file': 'results/blind_judgement/score-researcher-supplement.json',
                                             'key': 'rows_remaining',
                                             'pass_key': 'controls_pass'},
                                    'row': 'words20',
                                    'version': 'SPC',
                                    'outcome': 'both differ'}},
             'dagger': False}}
APPROVED_FRAME = {'headers': ['Quantity',
                              'Value (middle half of f, s where counted)',
                              'Reason; tried by then / shown beside / compared later / not tried',
                              'Images using it (* effect compared)',
                              'When: before or after its output; seen then',
                              'Who judged the look (sheet used)'],
                  'titles': ['Decision note for works 1 and 2', 'Decision note, empty'],
                  # a text of the note is one line or a list of lines, the lines drawn one under the other (version 16)
                  'note': ["Middle half: Q3 − Q1 as % of the rule's range, counted and compared after seeing the records.",
                           # what each share is compared with, moved from the caption of figure 8 (version 16)
                           ["Compared with, work 1: the story's terms shuffled among its sentences (keep the terms, break "
                            "the sentences) and uniform draws (break both);",
                            "each keeps every sentence's term count; mean of 1,000."],
                           ["Compared with, work 2: the whole lexicon and the 23 works' words, each once (keep the row's "
                            "values, break the choice of words);",
                            # the same-size draws of the main comparison of work 2, and what the percentiles and
                            # the other maps keep, in the words of the reason of map B (version 17)
                            "same-size draws from them, 20 per writer, 10,000 times (S1); percentiles keep the words; "
                            "other maps: check sentences."],
                           # the time, by the standard of the line on the time of note_empty (version 15), its words
                           # read and pointing to S1 for the full standard (version 16), and its one exception with its
                           # ground (version 16)
                           ["When: “after” means the maker who decided had seen the chosen value's outputs, or been told "
                            "of them,",
                            "by the time the quantity was last fixed for the work (full standard in S1)."],
                           "Drawing: read as before; of its four drawings, only the series' gray had been seen.",
                           "Judged: whether the two authors' rows differ within one version, on the 80 px judging sheets; "
                           "versions not compared."],
                  # the lines under the empty template of S1, which leave the time and the judgement open (version 13)
                  'note_empty': ["Middle half: Q3 − Q1 as % of the rule's range; counted and compared before or after "
                                 "seeing the values: …",
                                 # what each share is compared with (version 15), with its weighting (version 16)
                                 "Compared with: set, unit, weighting (each once or by frequency), what it keeps / breaks: …",
                                 # the time and the mark of the outline (version 14); the outputs of the chosen value, in
                                 # two lines (version 16)
                                 ["When: by the time the quantity was last fixed for the work, had the maker who decided "
                                  "seen the chosen value's outputs,",
                                  "or been told of them: …"],
                                 "‡: before, but after seeing outputs of alternatives: …",
                                 "Judged: what the judges compared, and what they did not: …"],
                  'acted_on_labels': 'in the repository the column holds the image files; the numbers of the figures of the paper are '
                                     'put in when the manuscript is made, outside the repository',
                  'compared_mark': ' *',
                  'dagger': {'mark': '† ', 'meaning': 'claim weakened in the paper after the values were seen'}}
APPROVED_LAYOUT = {'width_mm': 177.8,
 'dpi': 300,
 'colour': 'black and white',
 'font_file': '/System/Library/Fonts/Helvetica.ttc',
 'column_share': {'quantity': 0.1, 'value': 0.19, 'reason': 0.19, 'acted_on': 0.19, 'when': 0.17, 'judged': 0.16},
 'max_words': 533}
# the files of the two judges (checks of the conditions)
MAIN_FILE = "results/blind_judgement/score.json"
ALSO_FILE = "results/blind_judgement/score-researcher-supplement.json"
# a text before a built phrase may not make a claim of its own: no clause of its own and none of these words
SAID = re.compile(r";|\b(?:rows?|words|range|ranges|agreed?|disagreed?|differ|differs|split|both|neither|all|G0|P|SPC)\b")
JUDGEMENT = "works/conditions/blind_judgement.json"  # its rows are the rows an answer may name
SHARE = re.compile(r"(\d+)%")
USED_QUANTITIES = ["f", "s"]
# used_range_rule of the conditions: the records, the phrase of the share and the sets each row may read are fixed here
USED_FILE = "results/paper_used_range/used_range.json"
USED_CONDITIONS = "works/conditions/paper_used_range.json"
SAME_FILE = "results/paper_work1_same_unit/same_unit.json"
SAME_CONDITIONS = "works/conditions/paper_work1_same_unit.json"
RANDOM_SET = "random_sentences"  # its share is the mean over the replicates, computed again from them
SHUF_FILE = "results/paper_work1_shuffled/shuffled.json"
SHUF_CONDITIONS = "works/conditions/paper_work1_shuffled.json"
SHUF_SCRIPT = "works/paper_work1_shuffled.py"  # its function shuffle shuffles the sentences again (version 11)
SHUF_SET = "shuffled_sentences"  # its share is the mean over the replicates, computed again from them
USED_PHRASE = "{text} ({n} {unit} {f}, {s}{beside})"
# the reference sets beside map B (version 13): read from their record, checked by its provenance, not computed again
REF_FILE = "results/paper_reference_range/reference_range.json"
REF_CONDITIONS = "works/conditions/paper_reference_range.json"
REF_SCRIPT = "works/paper_reference_range.py"
REF_N = {"lexicon_B": "lexicon_entries", "tokens_B": "reference_words"}  # the key of its n in checks.R1_constants
REF_INPUTS: set[str] = set()  # the inputs of the records of the sets beside, as read (versions 13 and 14)
# the distinct words of the 23 works beside map B and the percentiles (version 14): read from their record, checked by its
# provenance, not computed again
DIST_FILE = "results/paper_distinct_range/distinct_range.json"
DIST_CONDITIONS = "works/conditions/paper_distinct_range.json"
DIST_SCRIPT = "works/paper_distinct_range.py"
DIST_N = {"distinct_B": "reference_words", "distinct_P": "reference_words"}  # the key of its n in checks.D1_constants
# per record beside: its conditions, its script, the key of its constants in its checks and the key of the n of each set
BESIDE_RECORDS = {REF_FILE: (REF_CONDITIONS, REF_SCRIPT, "R1_constants", REF_N),
                  DIST_FILE: (DIST_CONDITIONS, DIST_SCRIPT, "D1_constants", DIST_N)}
# the number of the tiles of the row of the 20 words of the blind judgement, named in the last line of the note (the second up to version 14)
JUDGING_PX_KEY = "tiles.words20_px"
# the texts of the note (version 15; version 16): their heads in order, the heads of the texts of note_empty, the words of
# the standard of the time shared with note_empty, and the record of the drawing of work 1 that names what had been seen
# and what had not
NOTE_HEADS = ["Middle half", "Compared with, work 1", "Compared with, work 2", "When", "Drawing", "Judged"]
EMPTY_HEADS = ["Middle half", "Compared with", "When", "‡", "Judged"]
WHEN_STANDARD = ["by the time the quantity was last fixed for the work", "the maker who decided",
                 "seen the chosen value's outputs", "been told of them"]
ALL_HEADS = {*NOTE_HEADS, *EMPTY_HEADS}  # no line after the first of a text begins with any of these (version 16)
WHEN_POINTER = "full standard in S1"  # version 16
DRAWING_WORDS = ["read as before", "only the series' gray had been seen"]  # the ground of the exception (version 16)
EMPTY_WEIGHTING = "weighting (each once or by frequency)"  # version 16
# what the texts on works 1 and 2 must name of the sets beside (version 16): the words of the labels of work 1, the
# labels of work 2, and the records of the number of replicates of both sets beside of work 1
COMPARED_W1_WORDS = ["shuffled", "uniform draws"]
COMPARED_W2_WORDS = ["each once"]
REPLICATES_OF = [(SHUF_CONDITIONS, "random.replicates"), (SAME_FILE, "checks.W1_constants.replicates")]
# the same-size draws named in the text on work 2 (version 17): the constants of the record of the distinct words, the sets
# of its draws whose replicates are counted, and the phrases of the text against the reason of the row of map B
DRAWS_CONSTANTS = {"per_writer": "checks.D1_constants.words_per_author", "times": "checks.D1_constants.replicates",
                   "authors": "checks.D1_constants.authors", "tiles": "checks.D1_constants.tiles"}
DRAWS_SETS = ["sets.draws_B.replicates", "sets.draws_P.replicates"]
DRAWS_PHRASES = {"per_writer": "{} per writer", "times": "{:,} times"}
COMPARED_W2_KEEP = ["percentiles keep the words", "other maps: check sentences"]
MAP_B_REASON = ("w2_map_b", "two other maps, on check sentences")
NOT_SAID = "other maps keep the words"  # the phrase of version 16 that contradicted the reason of map B, any case
# the numbers of the draws in any other text, written loosely (after the review of version 17): the number of words per
# writer with up to one word before 'per writer', and the replicates with or without a separator of thousands
LOOSE_SEP = r"[,\s\u2009\u202f]?"
DRAWING_COND = "works/conditions/work1_drawing.json"
# what was seen comes between SEEN_HEAD and NOT_SEEN_HEAD, what was not after NOT_SEEN_HEAD (case kept: "Not seen
# before:" holds "seen before:")
SEEN_HEAD, NOT_SEEN_HEAD = "Seen before:", "Not seen before:"
DRAWING_SEEN = {"seen": "common.png", "not_seen": "any tile of work 1 lit or stretched"}
# the shares beside, in order: (record, set, label); a share beside in the record of the row has no file in the conditions
# (the form of version 12 and before; from version 13 no row uses it)
USED_SETS = {"w1_lexicon": (SAME_FILE, SAME_CONDITIONS, "sentences",
                            [(SHUF_FILE, SHUF_SET, "terms shuffled among them"),
                             (SAME_FILE, RANDOM_SET, "uniform draws, same term counts")]),
             "w2_map_b": (USED_FILE, USED_CONDITIONS, "main_G0.pooled",
                          [(REF_FILE, "lexicon_B", "whole lexicon"), (DIST_FILE, "distinct_B", "23 works' words")]),
             "w2_percentiles": (USED_FILE, USED_CONDITIONS, "main_P.pooled", [(DIST_FILE, "distinct_P", "23 works' words")])}
# the values the quartiles of every set shown are computed again from (after the review of Codex on version 8)
WORK1_FILE = "results/work1/work1.json"
WORDS_FILE = "results/work2_words/words.json"
VALUES_FILE = "results/paper_words_values/words_values.json"
VALUES_CONDITIONS = "works/conditions/paper_words_values.json"
SAME_SCRIPT = "works/paper_work1_same_unit.py"  # its function draw draws the random sentences again (version 10)
WORK1_CONDITIONS = "works/conditions/work1.json"  # the lexicon of work 1
RECHECK_FILES = [WORK1_FILE, WORDS_FILE, VALUES_FILE, VALUES_CONDITIONS, USED_CONDITIONS, SAME_CONDITIONS, SAME_SCRIPT,
                 WORK1_CONDITIONS, SHUF_FILE, SHUF_CONDITIONS, SHUF_SCRIPT, REF_FILE, REF_CONDITIONS, REF_SCRIPT,
                 DIST_FILE, DIST_CONDITIONS, DIST_SCRIPT]
PERCENT = re.compile(r"(?<![\d.])(?:over )?\d+(?:\.\d+)?%")
# any way of writing a share outside the built phrases (checks of the conditions)
ANY_SHARE = re.compile(r"[%％]|\bper\s*cent\b|\bpercent(?:age)?s?\b", re.IGNORECASE)


def note_texts(x: list, heads: list[str], where: str) -> tuple[list[str], list[str]]:
    """The texts of a note, each one line or a list of lines (version 16), joined by single spaces, and its lines in
    order; no line after the first of a text begins with a head."""
    texts, lines = [], []
    for t in x:
        ls = [t] if isinstance(t, str) else t
        if (not isinstance(ls, list) or not ls
                or not all(isinstance(ln, str) and ln and ln == ln.strip() for ln in ls)
                or any(ln.split(":", 1)[0] in ALL_HEADS or ln.startswith(tuple(heads)) for ln in ls[1:])):
            raise SystemExit(f"{where}: a text is not one line or a list of lines, or a line after its first has a head")
        texts.append(" ".join(ls))
        lines.extend(ls)
    return texts, lines


def number(x, where: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise SystemExit(f"{where}: {x!r} is not a finite number")
    return float(x)
MIN_FONT_PT = 7.0  # the check of the conditions
# Drawing choices the conditions leave open (decision 1 of .steering/20261028-decision-note)
FONT_PT = 8.0
FONT_INDEX = {"regular": 0, "bold": 1}  # faces of Helvetica.ttc
PAD_PT = 3.0
RULE_PX = 2
TITLE_GAP_PT = 4.0
EMPTY_ROWS = 3
EMPTY_ROW_LINES = 2
# English title case: these words stay lower case unless first or last
MINOR_WORDS = {"a", "an", "and", "as", "at", "but", "by", "for", "in", "nor", "of", "on", "or", "the", "to"}
PLACEHOLDER = re.compile(r"\{([^{}]+)\}")
TITLE_CASE = " in title case"


def title_case(s: str) -> str:
    words = s.lower().split()
    return " ".join(w if 0 < i < len(words) - 1 and w in MINOR_WORDS else w[:1].upper() + w[1:]
                    for i, w in enumerate(words))


def read_key(c: dict, key: str, where: str):
    v = c
    for part in key.split("."):
        if not isinstance(v, dict) or part not in v:
            raise SystemExit(f"{where}: key {key} does not exist")
        v = v[part]
    return v


def as_text(v, where: str) -> str:
    if isinstance(v, bool) or not isinstance(v, (str, int, float)):
        raise SystemExit(f"{where}: {v!r} is not a string or a number")
    return v if isinstance(v, str) else repr(v)  # numbers as stored, no rounding


def load_json(rel: str, cache: dict) -> dict:
    """A JSON file read once: the bytes compared with HEAD are the bytes parsed, and no FILL mark may remain."""
    if rel not in cache:
        raw = (ROOT / rel).read_bytes()
        head = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True)
        if head.returncode != 0 or head.stdout != raw:
            raise SystemExit(f"{rel} differs from its committed version (or is not committed)")
        c = json.loads(raw)
        left = cm.placeholders(c)
        if left:
            raise SystemExit(f"{rel}: inputs still to be filled in: {', '.join(left)}")
        cache[rel] = c
    return cache[rel]


def strings(x, where: str, what: str) -> list[str]:
    if not isinstance(x, list) or not x or not all(isinstance(s, str) and s for s in x):
        raise SystemExit(f"{where}: {what} is not a non-empty list of non-empty strings")
    return x


def resolve_value(v: dict, cache: dict, where: str, approved: str | None = None, rid: str = "") -> tuple[str, dict]:
    """The text of a value and what was read for it; a text before a built phrase must be the approved one."""
    if set(v) in ({"file", "keys", "format"}, {"file", "keys", "format", "used_range"}):
        if approved is not None:
            raise SystemExit(f"{where}: a text is fixed in the script for this cell but the cell is a format")
        if not isinstance(v["format"], str) or not PLACEHOLDER.search(v["format"]):
            raise SystemExit(f"{where}: the format shows no key")
        c = load_json(v["file"], cache)
        read = {k: read_key(c, k, where) for k in strings(v["keys"], where, "keys")}
        by_name = {}
        for k, x in read.items():
            name = k.rsplit(".", 1)[-1]
            if name in by_name:
                raise SystemExit(f"{where}: two keys end in {name}")
            by_name[name] = x
        used = set()

        def fill(m: re.Match) -> str:
            name = m.group(1)
            case = name.endswith(TITLE_CASE)
            base = name[: -len(TITLE_CASE)] if case else name
            if base not in by_name:
                raise SystemExit(f"{where}: {{{name}}} names no key read")
            used.add(base)
            t = as_text(by_name[base], where)
            if case and not isinstance(by_name[base], str):
                raise SystemExit(f"{where}: {{{name}}} is not a string")
            return title_case(t) if case else t

        text = PLACEHOLDER.sub(fill, v["format"])
        if used != set(by_name):
            raise SystemExit(f"{where}: keys read but not shown: {sorted(set(by_name) - used)}")
        if "{" in text or "}" in text:
            raise SystemExit(f"{where}: an unmatched brace is left in {text!r}")
        if ANY_SHARE.search(text) or any(mk in text for mk in ("†", "*")):
            raise SystemExit(f"{where}: a reserved mark or a percentage in the format {text!r}")
        if "used_range" in v:
            u, text = check_used_range(v["used_range"], text, cache, where, rid)
            return text, {"file": v["file"], "read": read, "used_range": u}
        return text, {"file": v["file"], "read": read}
    if set(v) in ({"literal"}, {"literal", "check"}, {"text", "check"}, {"text", "check", "used_range"},
                  {"text", "check", "answers"}):
        base = v["literal"] if "literal" in v else v["text"]
        if not isinstance(base, str) or not base:
            raise SystemExit(f"{where}: the literal or text is not a non-empty string")
        if "literal" in v and PERCENT.search(base):
            raise SystemExit(f"{where}: a percentage in a literal (required_checks)")
        extra, text = {}, base
        if "used_range" in v:
            extra["used_range"], text = check_used_range(v["used_range"], base, cache, where, rid)
        if "answers" in v:
            extra["answers"], text = check_answers(v["answers"], base, cache, where)
        if any(mk in base for mk in ("†", "*")) or PERCENT.search(base):
            raise SystemExit(f"{where}: a reserved mark or a percentage in the text {base!r}")
        if "text" in v and base != approved:
            raise SystemExit(f"{where}: the text {base!r} is not the one fixed in the script ({approved!r})")
        if "text" not in v and approved is not None:
            raise SystemExit(f"{where}: a text is fixed in the script for this cell but the cell has none")
        if "text" in v and SAID.search(base):
            raise SystemExit(f"{where}: the text {base!r} before the built phrase makes a claim of its own")
        if "check" not in v:
            return text, {"literal": True, **extra}
        ch = v["check"]
        if not isinstance(ch, dict) or set(ch) != {"file", "keys", "must_contain"}:
            raise SystemExit(f"{where}: the check needs a file, keys and must_contain")
        must = strings(ch["must_contain"], where, "must_contain")
        c = load_json(ch["file"], cache)
        read = {k: as_text(read_key(c, k, where), where) for k in strings(ch["keys"], where, "keys")}
        missing = [s for s in must if not any(s in x for x in read.values())]
        if missing:
            raise SystemExit(f"{where}: not found in {ch['keys']} of {ch['file']}: {missing}")
        return text, {"literal": True, "check": {"file": ch["file"], "read": read, "found": ch["must_contain"]}, **extra}
    raise SystemExit(f"{where}: the value is neither a file with keys and format nor a literal")


def rule_ranges() -> dict[str, tuple[float, float]]:
    """The values the rule can give to f and s, with p and q at the ends of [0, 1] (used_range_rule)."""
    from atact import mapping as m

    return {"f": (m.F0, m.F0 + m.F1), "s": (m.S0, m.S0 + m.S1)}


def share_of(rec: dict, set_name: str, n: int, where: str) -> dict[str, float]:
    """The middle half of f and s of one set of the record, in percent of the range of the rule (used_range_rule)."""
    ranges = rule_ranges()
    out = {}
    for q in USED_QUANTITIES:
        t = read_key(rec, f"sets.{set_name}.{q}", where)
        if not isinstance(t, dict):
            raise SystemExit(f"{where}: sets.{set_name}.{q} is not an object")
        if t.get("n") != n or isinstance(t.get("n"), bool):
            raise SystemExit(f"{where}: sets.{set_name}.{q} counts {t.get('n')!r} values, not {n}")
        q1, q3, lo, hi, rec_share = (number(read_key(t, k, where), where)
                                     for k in ("q1", "q3", "min", "max", "share_quartiles_percent"))
        if not lo <= q1 <= q3 <= hi:
            raise SystemExit(f"{where}: sets.{set_name}.{q} is not ordered min <= q1 <= q3 <= max")
        a, b = ranges[q]
        out[q] = 100 * (q3 - q1) / (b - a)
        if abs(out[q] - rec_share) > 1e-9:
            raise SystemExit(f"{where}: the share of sets.{set_name}.{q} is not 100 (q3 - q1) / width")
    return out


def shown_matches(shown, shares: dict[str, float], where: str) -> None:
    if not isinstance(shown, dict) or list(shown) != USED_QUANTITIES:
        raise SystemExit(f"{where}: shown is not one string for each of {USED_QUANTITIES}")
    for q in USED_QUANTITIES:
        m = SHARE.fullmatch(shown[q]) if isinstance(shown[q], str) else None
        if m is None:
            raise SystemExit(f"{where}: shown.{q} {shown[q]!r} is not 'N%'")
        if abs(shares[q] - math.floor(shares[q]) - 0.5) < 1e-6 or round(shares[q]) != int(m.group(1)):
            raise SystemExit(f"{where}: {q} spans {shares[q]:.2f}% of its range, which is not {shown[q]}")


def quantiles_of(xs: list, where: str) -> dict:
    import numpy as np

    a = np.array([number(x, where) for x in xs])
    q1, q3 = (float(x) for x in np.quantile(a, [0.25, 0.75], method="linear"))
    return {"min": float(a.min()), "q1": q1, "q3": q3, "max": float(a.max()), "n": int(a.size)}


def recheck(rec: dict, set_name: str, cache: dict, where: str) -> None:
    """The quartiles of every set shown are computed again from its values, sentence by sentence or word by word
    (used_range_rule; the words of the percentiles from results/paper_words_values, after the review of Codex on
    version 8)."""
    if set_name == "sentences":
        sents = read_key(load_json(WORK1_FILE, cache), "sentences", where)
        sources = [{q: [x[q] for x in sents] for q in USED_QUANTITIES}]
    else:
        vals = load_json(VALUES_FILE, cache)
        meta = read_key(vals, "meta", where)
        load_json(VALUES_CONDITIONS, cache)
        if (not isinstance(meta, dict) or meta.get("conditions_sha256") != cm.sha256(ROOT / VALUES_CONDITIONS)
                or meta.get("git_dirty") is not False or meta.get("library_sha256") != cm.library_sha256()
                or read_key(meta, "inputs_sha256.used_range", where) != cm.sha256(ROOT / USED_FILE)
                or read_key(meta, "inputs_sha256.words", where) != cm.sha256(ROOT / WORDS_FILE)):
            raise SystemExit(f"{where}: {VALUES_FILE} is not the record of {VALUES_CONDITIONS} made on a clean tree from {USED_FILE}")
        rows = read_key(vals, "rows", where)
        authors = read_key(load_json(WORDS_FILE, cache), "authors", where)
        src = [(a, w) for a, x in authors.items() for w in x["words"]]
        if (not isinstance(rows, list) or len(rows) != len(src)
                or any((x.get("author"), x.get("word")) != (a, w["word"])
                       or x.get("map_B") != {k: w[k] for k in ("V", "A", "f", "s")} for x, (a, w) in zip(rows, src))):
            raise SystemExit(f"{where}: the rows of {VALUES_FILE} are not, one to one and in order, the words of {WORDS_FILE}")
        version = {"main_G0.pooled": "map_B", "main_P.pooled": "percentiles"}[set_name]
        sources = [{q: [read_key(x, f"{version}.{q}", where) for x in rows] for q in USED_QUANTITIES}]
        if set_name == "main_G0.pooled":
            sources.append({q: [w[q] for a in authors.values() for w in a["words"]] for q in USED_QUANTITIES})
    for per in sources:
        for q, xs in per.items():
            got = quantiles_of(xs, where)
            want = {k: read_key(rec, f"sets.{set_name}.{q}.{k}", where) for k in got}
            if got != want:
                raise SystemExit(f"{where}: sets.{set_name}.{q} of the record is not computed from the values: {got} {want}")


def sentences_z(cache: dict, where: str) -> list[dict]:
    """The sentences of work 1, each with a positive number of matched terms Z equal to its matched terms (checks of the
    conditions, version 12), before any of them is drawn or shuffled again."""
    sents = read_key(load_json(WORK1_FILE, cache), "sentences", where)
    if not isinstance(sents, list) or not sents or any(
            not isinstance(p, dict) or not isinstance(p.get("matched"), list) or isinstance(p.get("Z"), bool)
            or not isinstance(p.get("Z"), int) or p["Z"] < 1 or p["Z"] != len(p["matched"]) for p in sents):
        raise SystemExit(f"{where}: a sentence of {WORK1_FILE} has no matched terms or a Z other than their number")
    return sents


def redraw(rec: dict, cache: dict, where: str) -> None:
    """Every replicate of the random sentences drawn again by the committed function of the record, with the seed and the
    number of replicates of its conditions, and compared with the record exactly (used_range_rule, version 10)."""
    import importlib

    from atact.lexicon import load_lexicon

    cm.same_as_head([ROOT / SAME_SCRIPT])  # the committed bytes, before they are imported
    if read_key(rec, "meta.script_sha256", where) != cm.sha256(ROOT / SAME_SCRIPT):
        raise SystemExit(f"{where}: {SAME_SCRIPT} is not the script of {SAME_FILE}")
    rnd = read_key(load_json(SAME_CONDITIONS, cache), "random", where)
    seed, reps = rnd.get("seed"), rnd.get("replicates")
    if read_key(rec, "checks.W1_constants", where) != {**read_key(load_json(SAME_CONDITIONS, cache), "constants", where)} \
            or not isinstance(seed, int) or not isinstance(reps, int):
        raise SystemExit(f"{where}: the constants of {SAME_FILE} are not those of {SAME_CONDITIONS}")
    lexc = read_key(load_json(WORK1_CONDITIONS, cache), "lexicon", where)
    nrc = load_lexicon(ROOT / lexc["file"])
    if nrc.sha256 != lexc["sha256"] or nrc.sha256 != read_key(rec, "meta.nrc_sha256", where):
        raise SystemExit(f"{where}: the lexicon of work 1 is not the one of {SAME_FILE}")
    terms = sorted(nrc.entries)
    Z = [p["Z"] for p in sentences_z(cache, where)]
    width = {q: b - a for q, (a, b) in rule_ranges().items()}
    su = importlib.import_module("paper_work1_same_unit")
    if Path(su.__file__).resolve() != (ROOT / SAME_SCRIPT).resolve():
        raise SystemExit(f"{where}: the module imported is {su.__file__}, not {SAME_SCRIPT}")
    got = su.draw(seed, reps, Z, [nrc.entries[x][0] for x in terms], [nrc.entries[x][1] for x in terms], width)
    if got != read_key(rec, f"{RANDOM_SET}.replicates", where):
        raise SystemExit(f"{where}: the random sentences drawn again differ from {SAME_FILE}")


def random_share(rec: dict, n: int, where: str, cache: dict) -> dict[str, float]:
    """The mean over the replicates of the middle half of the random sentences, computed again from the replicates of
    the record, each drawn again (used_range_rule)."""
    import numpy as np

    redraw(rec, cache, where)
    out = {}
    replicates = read_key(rec, "checks.W1_constants.replicates", where)
    if read_key(rec, "checks.W1_constants.sentences", where) != n:
        raise SystemExit(f"{where}: the random sentences are not {n} per replicate")
    for q in USED_QUANTITIES:
        xs = read_key(rec, f"{RANDOM_SET}.replicates.{q}.share_quartiles_percent", where)
        if not isinstance(xs, list) or isinstance(replicates, bool) or len(xs) != replicates or replicates < 2:
            raise SystemExit(f"{where}: {RANDOM_SET}.replicates.{q} is not one share for each of the replicates")
        mean = float(np.asarray([number(x, where) for x in xs]).mean())
        if mean != number(read_key(rec, f"{RANDOM_SET}.summary.{q}.share_quartiles_percent.mean", where), where):
            raise SystemExit(f"{where}: the mean of {RANDOM_SET}.{q} is not computed from its replicates")
        out[q] = mean
    return out


def shuffled_share(n: int, cache: dict, where: str) -> dict[str, float]:
    """The mean over the replicates of the middle half of the story's own matched terms shuffled among its sentences,
    computed again from the replicates of the record, each shuffled again by the committed function of the record with
    the seed and the number of replicates of its conditions (used_range_rule, version 11)."""
    import importlib

    import numpy as np
    from atact.lexicon import load_lexicon

    rec = load_json(SHUF_FILE, cache)
    cond = load_json(SHUF_CONDITIONS, cache)
    meta = read_key(rec, "meta", where)
    cm.same_as_head([ROOT / SHUF_SCRIPT, ROOT / SAME_SCRIPT])  # the committed bytes, before they are imported
    made = {"work1": WORK1_FILE, "work1_conditions": WORK1_CONDITIONS, "same_unit": SAME_FILE,
            "same_unit_conditions": SAME_CONDITIONS, "same_unit_script": SAME_SCRIPT}
    if (not isinstance(meta, dict) or meta.get("conditions_sha256") != cm.sha256(ROOT / SHUF_CONDITIONS)
            or meta.get("git_dirty") is not False or meta.get("library_sha256") != cm.library_sha256()
            or meta.get("script_sha256") != cm.sha256(ROOT / SHUF_SCRIPT)
            or set(read_key(meta, "inputs_sha256", where)) != set(made)
            or any(read_key(meta, f"inputs_sha256.{k}", where) != cm.sha256(ROOT / f) for k, f in made.items())):
        raise SystemExit(f"{where}: {SHUF_FILE} is not the record of {SHUF_CONDITIONS} made on a clean tree by {SHUF_SCRIPT} "
                         f"from {sorted(made.values())}")
    if read_key(rec, "rule_range", where) != {q: list(v) for q, v in rule_ranges().items()}:
        raise SystemExit(f"{where}: the range of the rule of {SHUF_FILE} is not the installed one")
    if read_key(rec, "sentences", where) != read_key(load_json(SAME_FILE, cache), "sets.sentences", where):
        raise SystemExit(f"{where}: the sentences of {SHUF_FILE} are not those of {SAME_FILE}")
    k = read_key(cond, "constants", where)
    seed, reps = read_key(cond, "random.seed", where), read_key(cond, "random.replicates", where)
    if (read_key(rec, "checks.X1_constants", where) != k or k.get("sentences") != n or k.get("seed") != seed
            or k.get("replicates") != reps or not isinstance(seed, int) or isinstance(reps, bool)
            or not isinstance(reps, int) or reps < 2):
        raise SystemExit(f"{where}: the constants of {SHUF_FILE} are not those of {SHUF_CONDITIONS} for {n} sentences")
    lexc = read_key(load_json(WORK1_CONDITIONS, cache), "lexicon", where)
    nrc = load_lexicon(ROOT / lexc["file"])
    if nrc.sha256 != lexc["sha256"] or nrc.sha256 != read_key(meta, "nrc_sha256", where):
        raise SystemExit(f"{where}: the lexicon of work 1 is not the one of {SHUF_FILE}")
    sents = sentences_z(cache, where)
    tokens = [t for p in sents for t in p["matched"]]
    Z = [p["Z"] for p in sents]
    width = {q: b - a for q, (a, b) in rule_ranges().items()}
    # importing runs the top of each script, which sets sys.pycache_prefix to a new temporary directory: restore it
    prefix = sys.pycache_prefix
    try:
        sh = importlib.import_module("paper_work1_shuffled")
        su = importlib.import_module("paper_work1_same_unit")
    finally:
        made_prefix, sys.pycache_prefix = sys.pycache_prefix, prefix
        if made_prefix != prefix and made_prefix and Path(made_prefix).is_dir():
            shutil.rmtree(made_prefix, ignore_errors=True)
    for mod, script in ((sh, SHUF_SCRIPT), (su, SAME_SCRIPT)):
        if Path(mod.__file__).resolve() != (ROOT / script).resolve():
            raise SystemExit(f"{where}: the module imported is {mod.__file__}, not {script}")
    got = sh.shuffle(seed, reps, tokens, Z, nrc.entries, width, su)
    if got.pop("x3") != reps:  # X3 of the record: every replicate checked to be a permutation
        raise SystemExit(f"{where}: not every replicate shuffled again was checked as a permutation")
    if got != read_key(rec, f"{SHUF_SET}.replicates", where):
        raise SystemExit(f"{where}: the sentences shuffled again differ from {SHUF_FILE}")
    out = {}
    for q in USED_QUANTITIES:
        xs = read_key(rec, f"{SHUF_SET}.replicates.{q}.share_quartiles_percent", where)
        if not isinstance(xs, list) or len(xs) != reps:
            raise SystemExit(f"{where}: {SHUF_SET}.replicates.{q} is not one share for each of the replicates")
        mean = float(np.asarray([number(x, where) for x in xs]).mean())
        if mean != number(read_key(rec, f"{SHUF_SET}.summary.{q}.share_quartiles_percent.mean", where), where):
            raise SystemExit(f"{where}: the mean of {SHUF_SET}.{q} is not computed from its replicates")
        out[q] = mean
    return out


def reference_share(file: str, set_name: str, cache: dict, where: str) -> dict[str, float]:
    """The middle half of f and s of a set beside (the reference sets beside map B, version 13; the distinct words of the
    23 works beside map B and the percentiles, version 14), read from its record and checked by its provenance (its
    conditions, script and inputs at HEAD, a clean tree, the installed library and range of the rule, its n the count of
    its conditions) and as 100 (q3 - q1) / width; not computed again from the values, which need the whole lexicon and the
    words of the 23 works (the records checked their reconstruction: R2 to R4 and D2, D3)."""
    if file not in BESIDE_RECORDS:
        raise SystemExit(f"{where}: {file} is not a record beside fixed in the script")
    conditions, script, constants, n_key = BESIDE_RECORDS[file]
    rec = load_json(file, cache)
    cond = load_json(conditions, cache)
    meta = read_key(rec, "meta", where)
    cm.same_as_head([ROOT / script])
    inputs = {}
    for k, v in read_key(cond, "inputs", where).items():
        if k == "hashes":  # the text on how the inputs are checked, not an input
            continue
        if not isinstance(v, dict) or not isinstance(v.get("path"), str):
            raise SystemExit(f"{where}: the input {k!r} of {conditions} has no path")
        inputs[k] = v["path"]
    for p in inputs.values():
        if p.endswith(".json"):
            load_json(p, cache)  # the committed bytes at HEAD
        else:
            cm.same_as_head([ROOT / p])  # a script named as an input (version 14)
        REF_INPUTS.add(p)  # recorded among the files named in the output
    if (not isinstance(meta, dict) or meta.get("conditions_sha256") != cm.sha256(ROOT / conditions)
            or meta.get("git_dirty") is not False or meta.get("script_sha256") != cm.sha256(ROOT / script)):
        raise SystemExit(f"{where}: {file} is not the record of {conditions} made on a clean tree by {script}")
    if meta.get("library_sha256") != cm.library_sha256():
        raise SystemExit(f"{where}: the installed atact is not the library of {file}")
    if (not inputs or set(read_key(meta, "inputs_sha256", where)) != set(inputs)
            or any(read_key(meta, f"inputs_sha256.{k}", where) != cm.sha256(ROOT / p) for k, p in inputs.items())):
        raise SystemExit(f"{where}: {file} was not made from {sorted(inputs.values())} as committed")
    if read_key(rec, "rule_range", where) != {q: list(v) for q, v in rule_ranges().items()}:
        raise SystemExit(f"{where}: the range of the rule of {file} is not the installed one")
    if set_name not in n_key:
        raise SystemExit(f"{where}: {set_name!r} is not a set beside fixed in the script for {file}")
    n = read_key(rec, f"checks.{constants}.{n_key[set_name]}", where)
    if isinstance(n, bool) or not isinstance(n, int) or n < 2 or n != read_key(cond, f"constants.{n_key[set_name]}", where):
        raise SystemExit(f"{where}: checks.{constants}.{n_key[set_name]} of {file} is not the count of {conditions}")
    if file == DIST_FILE and read_key(rec, "checks.D3_weighted_tables_equal_reference_range", where) != 4:
        raise SystemExit(f"{where}: {file} did not recompute the four weighted sets of {REF_FILE} (D3)")
    return share_of(rec, set_name, n, where)


def check_used_range(u: dict, base: str, cache: dict, where: str, rid: str) -> tuple[dict, str]:
    """The middle half of f and s of the set of the record fixed for the row (and of each set beside, in order), checked against the
    strings shown, and the value built from it (used_range_rule)."""
    if not isinstance(u, dict) or not {"file", "set", "n", "unit", "shown"} <= set(u) <= {"file", "set", "n", "unit", "shown", "beside"}:
        raise SystemExit(f"{where}: used_range needs file, set, n, unit and shown (and may have beside)")
    if rid not in USED_SETS:
        raise SystemExit(f"{where}: no record of the share is fixed in the script for {rid!r}")
    file, conditions, main_set, beside_fixed = USED_SETS[rid]
    if u["file"] != file:
        raise SystemExit(f"{where}: used_range is not the record fixed in the script for {rid!r} ({file})")
    rec = load_json(u["file"], cache)
    meta = read_key(rec, "meta", where)
    load_json(conditions, cache)  # the bytes of the conditions of the record are those at HEAD
    if (not isinstance(meta, dict) or meta.get("conditions_sha256") != cm.sha256(ROOT / conditions)
            or meta.get("git_dirty") is not False):
        raise SystemExit(f"{where}: {u['file']} is not the record of {conditions} made on a clean tree")
    if cm.library_sha256() != meta.get("library_sha256"):
        raise SystemExit(f"{where}: the installed atact is not the library of {u['file']}")
    if u["file"] == SAME_FILE:  # the record of work 1 was made from the committed work1.json and used_range.json
        made = {"work1": WORK1_FILE, "used_range": USED_FILE}
        if any(read_key(meta, f"inputs_sha256.{k}", where) != cm.sha256(ROOT / f) for k, f in made.items()):
            raise SystemExit(f"{where}: {u['file']} was not made from {sorted(made.values())}")
    rule = read_key(rec, "rule_range", where)
    if rule != {q: list(v) for q, v in rule_ranges().items()}:
        raise SystemExit(f"{where}: the range of the rule of the record is not the installed one")
    n = u["n"]
    if isinstance(n, bool) or not isinstance(n, int) or not isinstance(u["unit"], str) or u["unit"] not in ("words", "sentences"):
        raise SystemExit(f"{where}: n or unit of used_range is not a count of words or sentences")
    if u["set"] != main_set:
        raise SystemExit(f"{where}: the set {u['set']!r} is not the one fixed for {rid!r} ({main_set!r})")
    shares = share_of(rec, main_set, n, where)
    recheck(rec, main_set, cache, where)
    shown_matches(u["shown"], shares, where)
    # one share beside in the record of the row (an object without file) or several in order (a list, each with its
    # record), as fixed in the script (version 11)
    b = u.get("beside", [])
    listed = isinstance(b, list)
    items = b if listed else [b]
    keys = {"file", "set", "label", "shown"} if listed else {"set", "label", "shown"}
    got_fixed = [(x.get("file") if listed else None, x.get("set"), x.get("label")) if isinstance(x, dict) and set(x) == keys
                 else None for x in items]
    # a list holds shares of other records (each with its file); an object, the one share of the record of the row; an
    # empty list written out is refused (the row has nothing beside)
    if (got_fixed != beside_fixed or (listed and not b and "beside" in u)
            or (b and listed != all(f is not None for f, _, _ in beside_fixed))):
        raise SystemExit(f"{where}: the shares beside are not the ones fixed for {rid!r} ({beside_fixed!r})")
    beside_txt, besides = "", []
    for x, (bfile, bset, blabel) in zip(items, beside_fixed):
        if bset == RANDOM_SET and bfile == SAME_FILE == u["file"]:
            beside = random_share(rec, n, where, cache)
        elif bset == SHUF_SET and bfile == SHUF_FILE:
            beside = shuffled_share(n, cache, where)
        elif bfile in BESIDE_RECORDS:
            beside = reference_share(bfile, bset, cache, where)
        elif bfile is None:
            beside = share_of(rec, bset, n, where)
            recheck(rec, bset, cache, where)
        else:
            raise SystemExit(f"{where}: no way of computing the share beside {bset!r} of {bfile!r} is fixed in the script")
        shown_matches(x["shown"], beside, where)
        beside_txt += f"; {blabel} {round(beside['f'])}%, {round(beside['s'])}%"
        besides.append({"file": bfile or u["file"], "set": bset, "shares_percent": beside})
    text = USED_PHRASE.format(text=base, n=n, unit=u["unit"], f=f"{round(shares['f'])}%", s=f"{round(shares['s'])}%",
                              beside=beside_txt)
    built = [f"{round(shares[q])}%" for q in USED_QUANTITIES] + [f"{round(x['shares_percent'][q])}%" for x in besides
                                                                for q in USED_QUANTITIES]
    return {"file": u["file"], "set": main_set, "shares_percent": shares, "beside": besides, "shown": u["shown"],
            "built": built}, text


def check_answers(a: dict, base: str, cache: dict, where: str) -> tuple[dict, str]:
    """The outcome of one row of one version under the two judges, against the one written, and the judged built from it
    (answers_rule)."""
    if not isinstance(a, dict) or set(a) != {"main", "also", "row", "version", "outcome"}:
        raise SystemExit(f"{where}: answers needs main, also, row, version and outcome")
    for k in ("row", "version", "outcome"):
        if not isinstance(a[k], str) or not a[k]:
            raise SystemExit(f"{where}: answers.{k} is not a non-empty string")
    if a["row"] not in strings(read_key(load_json(JUDGEMENT, cache), "rows", where), where, "rows"):
        raise SystemExit(f"{where}: {a['row']!r} is not a row of {JUDGEMENT}")
    if a["row"] not in ROW_PHRASE or a["outcome"] not in OUTCOME_PHRASE:
        raise SystemExit(f"{where}: no phrase is fixed for the row {a['row']!r} or the outcome {a['outcome']!r}")
    main, also = a["main"], a["also"]
    if not isinstance(main, dict) or set(main) != {"file", "key", "decided_key"}:
        raise SystemExit(f"{where}: answers.main needs file, key and decided_key")
    if not isinstance(also, dict) or set(also) != {"file", "key", "pass_key"}:
        raise SystemExit(f"{where}: answers.also needs file, key and pass_key")
    if (main["file"], also["file"]) != (MAIN_FILE, ALSO_FILE) or MAIN_FILE == ALSO_FILE:
        raise SystemExit(f"{where}: the files of the judges are not {MAIN_FILE} and {ALSO_FILE}")
    cm_ = load_json(main["file"], cache)
    ca = load_json(also["file"], cache)
    if read_key(cm_, main["decided_key"], where) is not True:
        raise SystemExit(f"{where}: {main['decided_key']} of {main['file']} is not true")
    if read_key(ca, also["pass_key"], where) is not True:
        raise SystemExit(f"{where}: {also['pass_key']} of {also['file']} is not true")
    kept = {}
    for name, c, d in (("main", cm_, main), ("also", ca, also)):
        lists = read_key(c, d["key"], where)
        rows = lists.get(a["version"]) if isinstance(lists, dict) else None
        if not isinstance(rows, list):
            raise SystemExit(f"{where}: {d['key']} of {d['file']} has no list for {a['version']}")
        kept[name] = a["row"] in rows
    outcome = "both differ" if all(kept.values()) else "neither" if not any(kept.values()) else "split"
    if outcome != a["outcome"]:
        raise SystemExit(f"{where}: the row {a['row']} of {a['version']} is {outcome!r} by the lists, not {a['outcome']!r}")
    text = f"{base}; for {ROW_PHRASE[a['row']]} {OUTCOME_PHRASE[outcome]}"
    return {"version": a["version"], "row": a["row"], "kept": kept, "outcome": outcome}, text


def wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    """Lines of at most width pixels, broken at spaces, and inside a path after a slash when a word is too long."""
    lines, line = [], ""
    for word in text.split():
        pieces = [word]
        if draw.textlength(word, font=font) > width:
            pieces = [p for p in re.split(r"(?<=/)", word) if p]
            if any(draw.textlength(p, font=font) > width for p in pieces):
                raise SystemExit(f"{word!r} does not fit in a column {width} px wide")
        for k, p in enumerate(pieces):
            sep = " " if line and k == 0 else ""
            if line and draw.textlength(line + sep + p, font=font) > width:
                lines.append(line)
                line = p
            else:
                line += sep + p
    if line:
        lines.append(line)
    return lines


def table(headers: list[str], rows: list, shares: list[float], width: int, fonts: dict, dpi: int,
          min_lines: int = 1) -> Image.Image:
    """A ruled table: a row is a list of cells, each a list of paragraphs (one per line of the record, e.g. one per
    image), or a string, drawn in bold as a band across the whole width."""
    pad = round(PAD_PT * dpi / 72)
    line_h = round(FONT_PT * 1.3 * dpi / 72)
    widths = [round(sh * width) for sh in shares]
    widths[-1] = width - sum(widths[:-1])
    scratch = ImageDraw.Draw(Image.new("L", (1, 1)))

    def lay(cells: list[list[str]], font) -> list[list[str]]:
        return [[ln for par in cell for ln in wrap(scratch, par, font, w - 2 * pad)] for cell, w in zip(cells, widths)]

    laid = [("cells", lay([[h] for h in headers], fonts["bold"]), True)]
    for r in rows:
        if isinstance(r, str):
            laid.append(("band", wrap(scratch, r, fonts["bold"], width - 2 * pad), True))
        else:
            laid.append(("cells", lay(r, fonts["regular"]), False))

    def n_lines(kind: str, x) -> int:
        return len(x) if kind == "band" else max(len(c) for c in x)

    heights = [max(min_lines if (i and kind == "cells") else 1, n_lines(kind, x)) * line_h + 2 * pad
               for i, (kind, x, _) in enumerate(laid)]
    img = Image.new("L", (width, sum(heights) + RULE_PX), 255)
    draw = ImageDraw.Draw(img)
    y = 0
    spans = []  # the vertical rules are drawn only through rows of cells
    for (kind, x, bold), h in zip(laid, heights):
        font = fonts["bold" if bold else "regular"]
        if kind == "band":
            for j, ln in enumerate(x):
                draw.text((pad, y + pad + j * line_h), ln, fill=0, font=font)
        else:
            cx = 0
            for cell, w in zip(x, widths):
                for j, ln in enumerate(cell):
                    draw.text((cx + pad, y + pad + j * line_h), ln, fill=0, font=font)
                cx += w
            spans.append((y, y + h))
        draw.rectangle((0, y, width - 1, y + RULE_PX - 1), fill=0)
        y += h
    draw.rectangle((0, y, width - 1, y + RULE_PX - 1), fill=0)
    bottom = img.height - 1
    for edge in (0, width - RULE_PX):  # the frame
        draw.rectangle((edge, 0, edge + RULE_PX - 1, bottom), fill=0)
    x = 0
    for w in widths[:-1]:
        x += w
        for y0, y1 in spans:
            draw.rectangle((x, y0, x + RULE_PX - 1, y1 + RULE_PX - 1), fill=0)
    return img


def title_image(text: str, width: int, fonts: dict, dpi: int, one_line: bool = False) -> Image.Image:
    line_h = round(FONT_PT * 1.3 * dpi / 72)
    scratch = ImageDraw.Draw(Image.new("L", (1, 1)))
    lines = wrap(scratch, text, fonts["bold"], width)
    if one_line and len(lines) != 1:
        raise SystemExit(f"{text!r} takes {len(lines)} lines, not one")
    img = Image.new("L", (width, len(lines) * line_h + round(TITLE_GAP_PT * dpi / 72)), 255)
    draw = ImageDraw.Draw(img)
    for j, ln in enumerate(lines):
        draw.text((0, j * line_h), ln, fill=0, font=fonts["bold"])
    return img


def md_table(headers: list[str], rows: list) -> str:
    def cell(s: str) -> str:
        return s.replace("|", "\\|")

    out = ["| " + " | ".join(cell(h) for h in headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        if isinstance(r, str):
            out.append("| **" + cell(r) + "** |" + " |" * (len(headers) - 1))
        else:
            out.append("| " + " | ".join(cell(x) for x in r) + " |")
    return "\n".join(out) + "\n"


def count_words(texts: list[str]) -> int:
    """Whitespace-separated tokens: a path is one word, and so is a mark standing alone."""
    return sum(len(t.split()) for t in texts)


def check_row(row: dict, where: str) -> None:
    if set(row) != ROW_FIELDS:
        raise SystemExit(f"{where}: fields {sorted(row)} are not {sorted(ROW_FIELDS)}")
    for k in ("id", "quantity", "when"):
        if not isinstance(row[k], str) or not row[k]:
            raise SystemExit(f"{where}.{k} is not a non-empty string")
    for k in ("acted_on", "compared"):
        x = row[k]
        if not isinstance(x, list) or not all(isinstance(p, str) and p for p in x):
            raise SystemExit(f"{where}.{k} is not a list of paths")
        if len(set(x)) != len(x):
            raise SystemExit(f"{where}.{k} names an image twice")
    if not row["acted_on"]:
        raise SystemExit(f"{where}.acted_on is empty")
    for p in row["acted_on"]:
        if not (ROOT / p).is_file():
            raise SystemExit(f"{where}.acted_on: {p} does not exist")
    stray = [p for p in row["compared"] if p not in row["acted_on"]]
    if stray:
        raise SystemExit(f"{where}.compared: {stray} are not in acted_on")
    if not isinstance(row["dagger"], bool):
        raise SystemExit(f"{where}.dagger is not true or false")
    r = row["reason"]
    if not isinstance(r, dict) or set(r) != {"why", "tried", "shown_beside", "later_compared", "not_tried"}:
        raise SystemExit(f"{where}.reason needs why, tried, shown_beside, later_compared and not_tried")
    if not isinstance(r["why"], str) or not r["why"]:
        raise SystemExit(f"{where}.reason.why is not a non-empty string")
    for k in ("tried", "shown_beside", "later_compared", "not_tried"):
        if not isinstance(r[k], list) or not all(isinstance(t, str) and t for t in r[k]):
            raise SystemExit(f"{where}.reason.{k} is not a list of non-empty strings")
    j = row["judged"]
    if isinstance(j, str):
        if j != NONE_RECORDED:
            raise SystemExit(f"{where}.judged: a string must be {NONE_RECORDED!r}; anything else is a literal")
    elif not isinstance(j, dict) or not ("literal" in j or "answers" in j):
        raise SystemExit(f"{where}.judged is neither {NONE_RECORDED!r}, a literal nor a text with answers")


def reason_text(r: dict) -> str:
    tried = "; ".join(r["tried"]) if r["tried"] else "none"
    beside = f" Shown beside: {'; '.join(r['shown_beside'])}." if r["shown_beside"] else ""
    later = f" Later compared with: {'; '.join(r['later_compared'])}." if r["later_compared"] else ""
    not_tried = f" Not tried: {'; '.join(r['not_tried'])}." if r["not_tried"] else ""
    return f"{r['why'][:1].upper()}{r['why'][1:]}. Tried: {tried}.{beside}{later}{not_tried}"


def main() -> None:
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} exists; refusing to overwrite")
    raw, c = cm.load_conditions(COND)
    if cm.need(c, "version", int) != VERSION:
        raise SystemExit(f"the script is for version {VERSION} of the conditions, not {c['version']}")
    cols = cm.need(c, "columns", list)
    ids = [col["id"] for col in cols]
    if ids != COLUMNS:
        raise SystemExit(f"the columns {ids} are not those the script draws {COLUMNS}")
    headers = [cm.need(col, "header", str, "columns.") for col in cols]
    lay = cm.need(c, "layout", dict)
    if lay.get("colour") != "black and white":
        raise SystemExit(f"layout.colour is {lay.get('colour')!r}; the script draws in black and white only")
    share = cm.need(lay, "column_share", dict, "layout.")
    if list(share) != COLUMNS or any(isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0
                                     for v in share.values()) or abs(sum(share.values()) - 1) > 1e-9:
        raise SystemExit(f"layout.column_share {share} is not a positive share of each column summing to 1")
    titles = cm.need(lay, "titles", list, "layout.")
    if len(titles) != 2 or not all(isinstance(t, str) and t for t in titles):
        raise SystemExit("layout.titles is not two non-empty strings")
    font_file = Path(cm.need(lay, "font_file", str, "layout."))
    dpi = cm.need(lay, "dpi", int, "layout.")
    width = round(cm.need(lay, "width_mm", float, "layout.") * dpi / MM_PER_INCH)
    label = cm.need(c, "acted_on_labels", str)
    mark_c = cm.need(c, "compared_mark", str)
    dag = cm.need(c, "dagger", dict)
    mark_d = cm.need(dag, "mark", str, "dagger.")
    note = cm.need(c, "note", list)
    texts, note_lines = note_texts(note, NOTE_HEADS, "note")  # version 16
    if not texts or any(mk.strip() in x for x in texts for mk in (mark_c, mark_d)):
        raise SystemExit("the note is not a list of texts, or a text holds a reserved mark")
    # the lines under the empty template (version 13): none a line of the note, and none naming a value or an outcome
    note_empty = cm.need(c, "note_empty", list)
    texts_empty, empty_lines = note_texts(note_empty, EMPTY_HEADS, "note_empty")  # version 16
    if (not texts_empty
            or any(mk.strip() in x for x in texts_empty for mk in (mark_c, mark_d))
            or set(texts_empty) & set(texts) or set(empty_lines) & set(note_lines)
            or any(PERCENT.search(x) for x in texts_empty)
            or any(p in x.casefold() for x in texts_empty
                   for p in (*OUTCOME_PHRASE.values(), *ROW_PHRASE.values(), "later", "after seeing the records",
                             "versions not compared"))):
        raise SystemExit("note_empty is not a list of lines of its own, or a line holds a mark, a value or an outcome")
    if not (any(x.startswith("When:") for x in texts_empty) and any(x.startswith("‡") for x in texts_empty)):
        raise SystemExit("note_empty has no line on the time or none on the mark ‡ (version 14)")
    compared_empty = [x for x in texts_empty if x.startswith("Compared with:")]
    if len(compared_empty) != 1 or EMPTY_WEIGHTING not in compared_empty[0]:
        raise SystemExit(f"note_empty has not one line on what each share is compared with naming {EMPTY_WEIGHTING!r} "
                         "(versions 15 and 16)")
    # the texts of the note in order (version 16), the text on the time holding the words of the standard of note_empty
    # (version 15) and pointing to S1 (version 16), the text on the exception holding its ground (version 16)
    if len(texts) != len(NOTE_HEADS) or not all(x.startswith(h + ":") for x, h in zip(texts, NOTE_HEADS)):
        raise SystemExit(f"the texts of the note do not begin, in order, with {NOTE_HEADS}")
    when_empty = [x for x in texts_empty if x.startswith("When:")]
    when_text = texts[NOTE_HEADS.index("When")]
    missing = [w for w in WHEN_STANDARD if w not in when_text or not all(w in x for x in when_empty)]
    if len(when_empty) != 1 or missing or WHEN_POINTER not in when_text:
        raise SystemExit(f"the text on the time of the note and that of note_empty do not both hold {missing}, or the "
                         f"note does not point to S1 ({WHEN_POINTER!r})")
    drawing_text = texts[NOTE_HEADS.index("Drawing")]
    if not all(w in drawing_text for w in DRAWING_WORDS):
        raise SystemExit(f"the text on the exception of the drawing does not hold {DRAWING_WORDS} (version 16)")
    max_words = cm.need(c, "max_words", int)
    frame = {"headers": headers, "titles": titles, "note": note, "note_empty": note_empty, "acted_on_labels": label,
             "compared_mark": mark_c, "dagger": dag}
    if frame != APPROVED_FRAME:
        raise SystemExit(f"the frame {[k for k in frame if frame[k] != APPROVED_FRAME[k]]} is not the one fixed in the script")
    fixed = {**{k: lay.get(k) for k in APPROVED_LAYOUT if k != "max_words"}, "max_words": max_words}
    if fixed != APPROVED_LAYOUT:
        raise SystemExit(f"the layout or words {[k for k in fixed if fixed[k] != APPROVED_LAYOUT[k]]} are not fixed in the script")
    size = round(FONT_PT * dpi / 72)
    if size * 72 / dpi < MIN_FONT_PT:
        raise SystemExit(f"the font of {size} px at {dpi} dpi is below {MIN_FONT_PT} pt")

    cache: dict = {}
    named: set[str] = set()
    resolved: dict[str, list] = {}
    if not set(APPROVED_CELLS) == set(APPROVED_ROWS) == {rid for ids in ROW_IDS.values() for rid in ids}:
        raise SystemExit("the cells fixed in the script are not those of the rows fixed in the script")
    for key in ROW_SETS:
        rows = cm.need(c, key, list)
        got = [r.get("id") if isinstance(r, dict) else None for r in rows]
        if got != ROW_IDS[key]:
            raise SystemExit(f"{key} has the rows {got}, not {ROW_IDS[key]}")
        resolved[key] = []
        for i, row in enumerate(rows):
            where = f"{key}[{i}]"
            check_row(row, where)
            for v, what in ((row["value"], "value"), (row["judged"], "judged")):
                if not isinstance(v, dict):
                    continue
                ch, u, a = v.get("check", {}), v.get("used_range", {}), v.get("answers", {})
                more = [d.get("file") if isinstance(d, dict) else None
                        for d in (ch, u, *((a.get("main"), a.get("also")) if isinstance(a, dict) else ()))]
                if isinstance(a, dict) and a:
                    more.append(JUDGEMENT)
                for f in (v.get("file"), *more):
                    if f is not None:
                        if not isinstance(f, str) or not (ROOT / f).is_file():
                            raise SystemExit(f"{where}.{what}: {f} does not exist")
                        named.add(f)
            named.update(row["acted_on"])
            if not isinstance(row["value"], dict):
                raise SystemExit(f"{where}.value is not an object")
            if "answers" in row["value"]:
                raise SystemExit(f"{where}.value: answers belong to judged")
            if isinstance(row["judged"], dict) and "used_range" in row["judged"]:
                raise SystemExit(f"{where}.judged: used_range belongs to the value")
            rid = row["id"]
            if row != APPROVED_ROWS[rid]:
                diff = [k for k in ROW_FIELDS if row.get(k) != APPROVED_ROWS[rid].get(k)]
                raise SystemExit(f"{where}: the fields {diff} are not the ones fixed in the script")
            text, read = resolve_value(row["value"], cache, where + ".value", APPROVED_TEXT.get((rid, "value")), rid)
            if isinstance(row["judged"], dict):
                if set(row["judged"]) not in ({"literal"}, {"literal", "check"}, {"text", "check", "answers"}):
                    raise SystemExit(f"{where}.judged is neither a literal nor a text with answers")
                judged, judged_read = resolve_value(row["judged"], cache, where + ".judged",
                                                    APPROVED_TEXT.get((rid, "judged")))
            else:
                if (rid, "judged") in APPROVED_TEXT:
                    raise SystemExit(f"{where}.judged: a text is fixed in the script for this cell but the cell has none")
                judged, judged_read = row["judged"], {"literal": True}
            reserved = [mk.strip() for mk in (mark_c, mark_d)]
            plain = [row["quantity"], text, reason_text(row["reason"]), *row["acted_on"], row["when"], judged]
            bad = [t for t in plain if any(mk in t for mk in reserved)]
            if bad:
                raise SystemExit(f"{where}: a reserved mark {reserved} is in {bad}")
            # a percentage only in the shares built from the record (checks of the conditions)
            built = read["used_range"]["built"] if "used_range" in read else []
            if ("used_range" in row["value"]) != (rid in USED_SETS):
                raise SystemExit(f"{where}: a share is required in the rows {sorted(USED_SETS)} and in no other")
            others = [row["quantity"], reason_text(row["reason"]), *row["acted_on"], row["when"], judged]
            rest = PERCENT.sub("", text) if built else text  # the value without the shares built from the record
            if PERCENT.findall(text) != built or any(ANY_SHARE.search(t) for t in (rest, *others)):
                raise SystemExit(f"{where}: a percentage other than the shares built from the record")
            v = row["value"]
            kind = "format" if "format" in v else "literal" if "literal" in v else "text"
            j = row["judged"]
            jkind = "none" if isinstance(j, str) else "literal" if "literal" in j else "text"
            written = {"quantity": row["quantity"],
                       "value": (kind, v.get("format", v.get("literal", v.get("text")))),
                       "reason": reason_text(row["reason"]), "when": row["when"],
                       "judged": (jkind, j if isinstance(j, str) else j.get("literal", j.get("text")))}
            if written != APPROVED_CELLS[rid]:
                diff = [k for k in written if written[k] != APPROVED_CELLS[rid][k]]
                raise SystemExit(f"{where}: the cells {diff} are not the ones fixed in the script")
            j = row["judged"]
            if isinstance(j, dict) and "literal" in j and SAID.search(j["literal"]):
                raise SystemExit(f"{where}.judged: a literal that says an outcome or names words (required_checks)")
            if key == "rows_work2" and (isinstance(j, dict) and "answers" in j) != (rid in VERSION_OF):
                raise SystemExit(f"{where}: answers are required in the rows {sorted(VERSION_OF)} and in no other")
            if isinstance(j, dict) and "answers" in j and VERSION_OF.get(rid) != j["answers"].get("version"):
                raise SystemExit(f"{where}: the version of the answers is not the version fixed for {rid!r}")
            resolved[key].append({
                "id": rid, "quantity": row["quantity"], "dagger": row["dagger"], "value": text, "reason": reason_text(row["reason"]),
                "acted_on": [p + (mark_c if p in row["compared"] else "") for p in row["acted_on"]],
                "when": row["when"], "judged": judged, "read": {"value": read, "judged": judged_read}})
    if not font_file.is_file():
        raise SystemExit(f"{font_file} does not exist")
    if {rid for rid, _ in APPROVED_TEXT} - {r["id"] for rs in resolved.values() for r in rs}:
        raise SystemExit("a text fixed in the script names a row that is not drawn")
    daggered = [r["quantity"] for rs in resolved.values() for r in rs if r["dagger"]]
    if daggered:
        raise SystemExit(f"rows {daggered} carry the dagger; version 15 has none in the main table")
    px = read_key(load_json(JUDGEMENT, cache), JUDGING_PX_KEY, "note")  # the images judged, named in the note (version 14)
    if (isinstance(px, bool) or not isinstance(px, int) or texts[-1].count(f" {px} px judging sheets") != 1
            or any(f" {px} px judging sheets" in x for x in texts[:-1])):
        raise SystemExit(f"the note does not name the {JUDGING_PX_KEY} of {JUDGEMENT} once")
    named.add(JUDGEMENT)
    # the 'only' of the judged of the seed: SPC is the one version naming the seed per spelling (version 14)
    vers = read_key(load_json(JUDGEMENT, cache), "versions", "w2_seed")
    if (not isinstance(vers, dict) or "seed per spelling" not in vers.get(VERSION_OF["w2_seed"], "")
            or any("seed per spelling" in v for k, v in vers.items() if k != VERSION_OF["w2_seed"])):
        raise SystemExit("w2_seed: the seed per spelling is not judged only in the version with the color")
    # the one exception to the line on the time: the drawing of work 1, before though one of its images had been seen
    # (version 15)
    drawn = [r for r in resolved["rows_work1"] if r["quantity"] == "drawing"]
    seen = drawn[0]["when"].removeprefix("before; seen: ") if len(drawn) == 1 else ""
    if len(drawn) != 1 or not drawn[0]["when"].startswith("before; seen: ") or not seen or seen == "none":
        raise SystemExit("work 1 has not one row of the drawing read as before with something seen (version 15)")
    told = read_key(load_json(DRAWING_COND, cache), "fixed_before_running", "note")
    a = told.find(SEEN_HEAD) if isinstance(told, str) else -1
    b = told.find(NOT_SEEN_HEAD, a) if a >= 0 else -1
    seen_part = told[a:b] if 0 <= a < b else ""
    not_seen_part = told[b:] if b >= 0 else ""
    if (not seen_part or DRAWING_SEEN["seen"] not in seen_part or DRAWING_SEEN["not_seen"] not in not_seen_part
            or DRAWING_SEEN["not_seen"] in seen_part or DRAWING_SEEN["seen"] in not_seen_part):
        raise SystemExit(f"{DRAWING_COND} does not name {DRAWING_SEEN['seen']!r} as seen and "
                         f"{DRAWING_SEEN['not_seen']!r} as not seen in its fixed_before_running")
    named.add(DRAWING_COND)
    # the texts on what each share is compared with (version 16): work 1 names its sets beside and their number of
    # replicates, read from both records; work 2 names the labels of the sets beside of its rows
    reps = [read_key(load_json(f, cache), k, "note") for f, k in REPLICATES_OF]
    w1_text, w2_text = texts[NOTE_HEADS.index("Compared with, work 1")], texts[NOTE_HEADS.index("Compared with, work 2")]
    w1_labels = [lab for _, _, lab in USED_SETS["w1_lexicon"][3]]
    w2_labels = sorted({lab for rid in ("w2_map_b", "w2_percentiles") for _, _, lab in USED_SETS[rid][3]})
    # the number of replicates as written, not followed by more digits (after the review of version 16)
    mean_of = (f"mean of {reps[0]:,}" if reps and not isinstance(reps[0], bool) and isinstance(reps[0], int) else "")
    mean_hits = len(re.findall(rf"(?<![\d,.])mean of {re.escape(mean_of[8:])}(?![\d,.]*\d)", w1_text)) if mean_of else 0
    if (any(isinstance(n, bool) or not isinstance(n, int) for n in reps) or len(set(reps)) != 1 or mean_hits != 1
            or any("mean of" in x for i, x in enumerate(texts) if i != NOTE_HEADS.index("Compared with, work 1"))
            or len(w1_labels) != len(COMPARED_W1_WORDS) or len(w2_labels) != 2
            or not all(w in lab and w in w1_text for w, lab in zip(COMPARED_W1_WORDS, w1_labels))
            or not all(lab in w2_text for lab in w2_labels) or not all(w in w2_text for w in COMPARED_W2_WORDS)):
        raise SystemExit(f"the texts on what each share is compared with do not name the sets beside {w1_labels}, "
                         f"{w2_labels} or the replicates {reps} (version 16)")
    named.update(f for f, _ in REPLICATES_OF)
    # the same-size draws of work 2 and what the percentiles and the other maps keep (version 17)
    dist = load_json(DIST_FILE, cache)
    dc = {k: read_key(dist, key, "note") for k, key in DRAWS_CONSTANTS.items()}
    sizes = [read_key(dist, key, "note") for key in DRAWS_SETS]
    if (any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in dc.values())
            or any(not isinstance(x, list) or len(x) != dc["times"] for x in sizes)
            or dc["authors"] * dc["per_writer"] != dc["tiles"]):
        raise SystemExit(f"{DIST_FILE}: the constants {dc} or the replicates of {DRAWS_SETS} do not agree (version 17)")
    drawn_phrases = {k: DRAWS_PHRASES[k].format(dc[k]) for k in DRAWS_PHRASES}
    i_w2 = NOTE_HEADS.index("Compared with, work 2")
    hits = {k: len(re.findall(rf"(?<![\d,.]){re.escape(p)}", w2_text)) for k, p in drawn_phrases.items()}
    loose = [rf"(?<![\d,.]){dc['per_writer']}\s*(?:\w+\s+)?per writer",
             rf"(?<![\d,.]){LOOSE_SEP.join(f'{dc["times"]:,}'.split(','))}(?![\d,.]*\d)\s*times"]
    others = [x for i, x in enumerate(texts) if i != i_w2] + list(texts_empty)  # note and note_empty (after the review)
    map_b = [r for r in resolved["rows_work2"] if r["id"] == MAP_B_REASON[0]]
    if (any(n != 1 for n in hits.values())
            or any(re.search(p, x, re.IGNORECASE) for x in others for p in loose)
            or not all(p in w2_text for p in COMPARED_W2_KEEP)
            or len(map_b) != 1 or MAP_B_REASON[1] not in map_b[0]["reason"]
            or any(NOT_SAID in x.casefold() for x in (*texts, *texts_empty))):
        raise SystemExit(f"the text on work 2 does not name {list(drawn_phrases.values())} once and in no other text, "
                         f"or {COMPARED_W2_KEEP} against the reason of {MAP_B_REASON[0]} {MAP_B_REASON[1]!r}, or holds "
                         f"{NOT_SAID!r} (version 17)")
    named.add(DIST_FILE)

    files = sorted(named)
    cm.same_as_head([COND, *(ROOT / p for p in CODE), *(ROOT / p for p in files)])
    files = sorted(set(files) | set(RECHECK_FILES) | REF_INPUTS)
    cm.same_as_head([ROOT / p for p in RECHECK_FILES])
    state = cm.git_state(["works", *files, "pyproject.toml", "uv.lock"])
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    def cells(r: dict) -> list[str]:
        return [r["quantity"], r["value"], r["reason"], r["acted_on"], r["when"], r["judged"]]

    filled_rows: list = []
    for key, band in ROW_SETS.items():
        filled_rows.append(band)
        filled_rows += [[[x] if isinstance(x, str) else x for x in cells(r)] for r in resolved[key]]

    fonts = {k: ImageFont.truetype(str(font_file), size, index=i) for k, i in FONT_INDEX.items()}
    shares = [share[k] for k in COLUMNS]
    parts = [title_image(titles[0], width, fonts, dpi), table(headers, filled_rows, shares, width, fonts, dpi)]
    parts += [Image.new("L", (width, round(TITLE_GAP_PT * dpi / 72)), 255),
              *(title_image(x, width, {"bold": fonts["regular"]}, dpi, one_line=True) for x in note_lines)]
    # the empty template is its own image, for supplement S1 (version 11), with lines of note under it (version 12), its
    # own general lines (version 13)
    empty_parts = [title_image(titles[1], width, fonts, dpi),
                   table(headers, [[[""]] * len(headers)] * EMPTY_ROWS, shares, width, fonts, dpi, min_lines=EMPTY_ROW_LINES),
                   Image.new("L", (width, round(TITLE_GAP_PT * dpi / 72)), 255),
                   *(title_image(x, width, {"bold": fonts["regular"]}, dpi, one_line=True) for x in empty_lines)]

    def stack(ps: list) -> Image.Image:
        out = Image.new("L", (width, sum(p.height for p in ps)), 255)
        y = 0
        for p in ps:
            out.paste(p, (0, y))
            y += p.height
        return out

    fig, empty = stack(parts), stack(empty_parts)
    drawn_filled = [titles[0], *headers, *ROW_SETS.values(), *note_lines,
                    *(t for rs in resolved.values() for r in rs for x in cells(r) for t in ([x] if isinstance(x, str) else x))]
    words = {"figure_8": count_words(drawn_filled), "empty_template_s1": count_words([titles[1], *headers, *empty_lines])}
    if words["figure_8"] > max_words:
        raise SystemExit(f"figure 8 has {words['figure_8']} words, more than {max_words}")

    md_rows: list = []
    for key, band in ROW_SETS.items():
        md_rows.append(band)
        md_rows += [[x if isinstance(x, str) else ", ".join(x) for x in cells(r)] for r in resolved[key]]
    md = (f"# {titles[0]}\n\n" + md_table(headers, md_rows) + "".join(f"\n{x}\n" for x in texts)
          + "\n" + label[0].upper() + label[1:] + ".\n"
          + f"\n# {titles[1]}\n\n" + md_table(headers, [[""] * len(headers)] * EMPTY_ROWS)
          + "".join(f"\n{x}\n" for x in texts_empty))

    OUT_DIR.parent.mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".decision_note-", dir=OUT_DIR.parent))
    try:
        for name, im in (("decision_note.png", fig), ("decision_note_empty.png", empty)):
            im.save(tmp / name, dpi=(dpi, dpi))
            with Image.open(tmp / name) as saved:
                got_dpi = tuple(round(d) for d in saved.info.get("dpi", (0, 0)))
                if saved.mode != "L" or saved.width != width or got_dpi != (dpi, dpi):
                    raise SystemExit(f"{name} is {saved.mode} {saved.size} at {got_dpi} dpi, not L {width} px wide at {dpi}")
        (tmp / "decision_note.md").write_text(md, encoding="utf-8")
        res = {"meta": {"conditions_sha256": hashlib.sha256(raw).hexdigest(),
                        "script_sha256": cm.sha256(Path(__file__)),
                        "common_sha256": cm.sha256(ROOT / "works" / "_common.py"), "font_sha256": cm.sha256(font_file),
                        "library_sha256": cm.library_sha256(),
                        **cm.versions("pillow"), **state, "started": started,
                        "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
               "files_named_sha256": {p: cm.sha256(ROOT / p) for p in files},
               **resolved,
               "dagger_rows": daggered,
               "judging_px": {"file": JUDGEMENT, "key": JUDGING_PX_KEY, "value": px},  # version 14
               "when_exception": {"row": drawn[0]["id"], "when": drawn[0]["when"], "file": DRAWING_COND,
                                  "key": "fixed_before_running", "names": DRAWING_SEEN,
                                  "seen_part": seen_part, "not_seen_part": not_seen_part},  # version 15
               "note_texts": {"note": texts, "note_empty": texts_empty, "lines_drawn": note_lines,
                              "empty_lines_drawn": empty_lines,
                              "checked": {"when_standard": WHEN_STANDARD, "when_pointer": WHEN_POINTER,
                                          "drawing_words": DRAWING_WORDS, "empty_weighting": EMPTY_WEIGHTING,
                                          "compared_w1_words": COMPARED_W1_WORDS, "w1_labels": w1_labels,
                                          "compared_w2_words": COMPARED_W2_WORDS, "w2_labels": w2_labels}},  # version 16
               "compared_replicates": {"records": [{"file": f, "key": k, "value": n} for (f, k), n in zip(REPLICATES_OF, reps)],
                                       "drawn": mean_of},  # version 16
               "compared_draws": {"file": DIST_FILE, "constants": [{"key": DRAWS_CONSTANTS[k], "value": v} for k, v in dc.items()],
                                  "replicates_counted": [{"key": key, "value": len(x)} for key, x in zip(DRAWS_SETS, sizes)],
                                  "drawn": list(drawn_phrases.values()), "hits_in_work2": hits,
                                  "not_elsewhere": {"patterns": loose, "texts": "the other texts of note and all of note_empty"},
                                  "keep": COMPARED_W2_KEEP,
                                  "map_b_reason": {"row": MAP_B_REASON[0], "holds": MAP_B_REASON[1]},
                                  "not_said": {"phrase": NOT_SAID, "how": "any case, as written; rewordings not sought"}},  # version 17
               "words": {**words, "max_words": max_words, "how": "whitespace-separated tokens, a path one word, "
                         "a mark standing alone one word; titles, headers, band labels, the note and the cells"},
               "drawing": {"width_px": width, "height_px": fig.height, "empty_height_px": empty.height, "mode": fig.mode,
                           "dpi": dpi, "font_pt": FONT_PT,
                           "column_share": share, "empty_rows": EMPTY_ROWS, "empty_row_lines": EMPTY_ROW_LINES,
                           "minor_words": sorted(MINOR_WORDS)},
               "outputs": {n: cm.sha256(tmp / n) for n in ("decision_note.png", "decision_note_empty.png", "decision_note.md")}}
        with (tmp / "decision_note.json").open("x", encoding="utf-8") as fh:
            fh.write(json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    except BaseException:
        shutil.rmtree(tmp)
        raise
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} appeared during the run; the output is left in {tmp}")
    tmp.rename(OUT_DIR)
    print(json.dumps({"words": words, **{k: [r["quantity"] for r in v] for k, v in resolved.items()}},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
