"""Figure 1 of the paper: a schematic of the rule and of the quantities it leaves open, from the conditions of
works/conditions/rule_schematic.json (version 6, committed before this script was written).

Usage:  uv run python works/rule_schematic.py     (refuses to overwrite; stops if an input is still "FILL")
Input:  the conditions; the files named in them (read at drawing time or checked against a literal); results/work1/
        common.png and work1.json for the one tile of work 1
Output: results/rule_schematic_v6/rule_schematic.png and rule_schematic.json (the SHA-256 of the conditions, every value
        read, every literal checked, the words, the smallest type)

Text and arrows only, plus one tile cut from an existing image (no resampling): six boxes joined by arrows; under the box
a quantity acts on, a label 'name: work 1 value / work 2 value'. Every text the figure draws is fixed in this script as
well as in the conditions (approved_text_rule): a difference stops the run.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.pycache_prefix = tempfile.mkdtemp(prefix="pycache-")  # never run a stale .pyc of src/ (ignored by git)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "works"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import _common as cm  # noqa: E402
import relief  # noqa: E402
import work1  # noqa: E402
from atact.field import procedural_field_separable  # noqa: E402

COND = ROOT / "works" / "conditions" / "rule_schematic.json"
OUT_DIR = ROOT / "results" / "rule_schematic_v6"
WORK1_COND = ROOT / "works" / "conditions" / "work1.json"
WORK1_RECORD = "results/work1/work1.json"
COMMON_PNG = "results/work1/common.png"
MM_PER_INCH = 25.4
VERSION = 6
MIN_FONT_PT = 7.0  # the check of the conditions
FONT_PX = 30  # 7.2 pt at 300 dpi
FONT_INDEX = {"regular": 0, "bold": 1}  # faces of Helvetica.ttc
CODE = ["works/rule_schematic.py", "works/_common.py", "works/work1.py", *("src/" + n for n in (
    "relief.py", "pipeline.py", "english_text.py", "japanese_text.py", "window.py", "stretch.py"))]
# the defaults of the installed generator that a check_absent may name (the text of the condition -> how to check it)
GENERATOR_DEFAULTS = {"atact.field: the default of warp is 0.0": ("RenderConfig", "warp", 0.0)}
WARP_FUNCTION = "procedural_field_separable"  # the generator work 1 draws its field with (src/relief.py, works/work1.py)
SPEC_PATH, SPEC_LITERAL = {"file", "key"}, {"literal"}  # the keys a value of a label may hold (a literal also one of the next)
SPEC_LITERAL_EXTRA = {"check", "check_absent", "note"}
CHECK_KEYS = {"must_contain": {"file", "keys", "must_contain"}, "equals": {"file", "key", "equals"}}
ABSENT_KEYS = {"file", "key", "default"}
NUMBER_WORDS = {4: "four"}
SEP = " / "  # between the value of work 1 and that of work 2 in a label

# approved_text_rule: what the conditions say and the figure draws, as fixed in this script (copied from version 6)
APPROVED = {'version': 6,
 'top_keys': ['boxes',
              'fixed_before_running',
              'layout',
              'not_claimed',
              'open_quantities',
              'outputs',
              'purpose',
              'tile',
              'version'],
 'layout': {'width_mm': 177.8,
            'dpi': 300,
            'colour': 'black, white and greys only (the figure must read in print without colour)',
            'font_file': '/System/Library/Fonts/Helvetica.ttc',
            'direction': 'left to right, one row of boxes joined by arrows'},
 'boxes': [{'id': 'unit', 'text': 'a unit of text (a sentence in work 1, a word in work 2)'},
           {'id': 'values', 'text': 'valence V and arousal A from a lexicon, put on a scale'},
           {'id': 'rule', 'text': 'the rule: V and A give fineness f, depth s and sharpness γ'},
           {'id': 'field', 'text': 'the field of the generator (a seed, four octaves)'},
           {'id': 'relief', 'text': 'the relief: height under a scale'},
           {'id': 'image', 'text': 'the image of the relief (how it is drawn)'}],
 'open_quantities': {'how': "labels in a second row under the box they act on, written as 'name: work 1 value / work "
                            "2 value'; each value is given either as a path {file, key} read at drawing time or as a "
                            "literal; a literal is checked against the key named in its 'check' (else the run "
                            'stops); a dash means the quantity does not act on the main images of that work; from '
                            'version 4 a literal may carry check_absent instead of check: the key named must be '
                            'absent from the file and the default named must hold in the installed generator (else '
                            'the run stops); from version 4 every check is explicit: must_contain (each string '
                            'occurs in the text of the keys named) or equals (the value of the key equals it), and a '
                            'literal with neither check nor check_absent is drawn as written (the literals of the '
                            'height scale and of the drawing and units of work 1)',
                     'work2_main': 'the main images of work 2 are those of works/conditions/work2_sp_color.json (its '
                                   'versions SP and SPC); the scoring of the blind judgement decides only which of '
                                   'the grey and the coloured images is shown as the main one, by a rule written in '
                                   "the conditions of the paper's figures before the scoring is read. Until then "
                                   'work2.image_shown is FILL',
                     'list': [{'name': 'lexicon',
                               'under': 'values',
                               'work1': {'file': 'works/conditions/work1.json', 'key': 'lexicon.name'},
                               'work2': {'literal': 'MEmoLon, Japanese',  # version 6
                                         'check': {'file': 'works/conditions/work2.json',
                                                   'key': 'lexicon.name',
                                                   'equals': 'MEmoLon, Japanese (MTL_grouped-ja.tsv)'}}},
                              {'name': 'scale of V and A',
                               'under': 'values',
                               'work1': {'literal': 'as rated (no map)'},
                               'work2': {'literal': 'percentiles among all uses of lexicon words in the 23 works, '
                                                    'each author weighted equally, spread over [-1, 1]',
                                         'check': {'file': 'works/conditions/work2_sp_color.json',
                                                   'keys': ['versions.SP', 'reference.what'],
                                                   'must_contain': ['2 p_valence - 1',
                                                                    '2 p_arousal - 1',
                                                                    'the 23 works',
                                                                    'each author weighing 1']}}},
                              {'name': 'which units',
                               'under': 'unit',
                               'work1': {'literal': 'every sentence'},
                               'work2': {'literal': '20 words per author',
                                         'check': {'file': 'results/paper_distinct_range/distinct_range.json',
                                                   'key': 'checks.D1_constants.words_per_author',
                                                   'equals': 20}}},
                              {'name': 'seed',
                               'under': 'field',
                               'work1': {'file': 'works/conditions/work1.json', 'key': 'relief.seed'},
                               'work2': {'literal': 'one per word from its spelling',
                                         'check': {'file': 'works/conditions/work2_sp_color.json',
                                                   'keys': ['seed'],
                                                   'must_contain': ['per word', 'SHA-256 of the UTF-8 word']}}},
                              {'name': 'band',
                               'under': 'field',
                               'work1': {'file': 'works/conditions/work1.json', 'key': 'relief.octaves'},
                               'work2': {'file': 'works/conditions/work2_words.json', 'key': 'field.octaves'}},
                              {'name': 'warp',
                               'under': 'field',
                               'work1': {'literal': 'none (generator default)',
                                         'check_absent': {'file': 'works/conditions/work1.json',
                                                          'key': 'relief.warp',
                                                          'default': 'atact.field: the default of warp is 0.0'}},
                               'work2': {'file': 'works/conditions/work2_words.json', 'key': 'field.warp'}},
                              {'name': 'height scale',
                               'under': 'relief',
                               'work1': {'literal': 'common to the series (per-tile min-max shown next to it)'},
                               'work2': {'literal': 'common'}},
                              {'name': 'smoothing parameter ε',
                               'under': 'relief',
                               'work1': {'literal': '-',
                                         'note': 'work 1 uses eps in the supplementary image common_eps.png and in the four '
                                                 'drawings of figure 4 (works/work1_drawing.py), not in its main images '
                                                 '(figures 2 and 3)'},
                               'work2': {'file': 'results/work2_sp_color/work2_sp_color.json', 'key': 'eps'}},
                              {'name': 'drawing',
                               'under': 'image',
                               'work1': {'literal': 'gray; light from two directions; stretched'},
                               'work2': {'literal': 'gray with color (OKLCH: lightness from the gray, chroma from '
                                                    'arousal, hue from valence)',
                                         'check': {'file': 'works/conditions/work2_sp_color.json',
                                                   'keys': ['versions.SPC', 'colour.hue.SPC.reads', 'colour.status'],
                                                   'must_contain': ['OKLCH',
                                                                    'lightness of its grey',
                                                                    'hue and chroma',
                                                                    'at p_valence 0',
                                                                    'saturation raises arousal']}}}]},
 'tile': {'what': 'the tile of the first sentence of work 1 under the common scale, cut from '
                  'results/work1/common.png at its place in the layout of work 1 (no resampling), shown under the '
                  "box 'image'",
          'check': 'the cut equals the tile drawn afresh by works/work1.py for sentence 1 byte for byte (else the '
                   'run stops)'},
 'outputs': 'results/rule_schematic_v6/rule_schematic.png and rule_schematic.json (provenance: the conditions read and '
            'their SHA-256, every value read and every literal checked); results/rule_schematic (version 5) is kept'}


# ---------------------------------------------------------------- reading values


def read_key(doc: dict, key: str, where: str):
    cur = doc
    for k in key.split("."):
        if not isinstance(cur, dict) or k not in cur:
            raise SystemExit(f"{where}: the key {key!r} is missing")
        cur = cur[k]
    return cur


def has_key(doc: dict, key: str, where: str) -> bool:
    """Whether the last key of the path is present; the parents must be there (a typo in them must not pass as absent)."""
    *parents, last = key.split(".")
    cur = doc
    for k in parents:
        if not isinstance(cur, dict) or k not in cur:
            raise SystemExit(f"{where}: the parent {k!r} of {key!r} is missing, so the key cannot be called absent")
        cur = cur[k]
    if not isinstance(cur, dict):
        raise SystemExit(f"{where}: the parent of {key!r} is not a mapping")
    return last in cur


class Files:
    """The JSON files the values are read from: committed bytes only, each read once, its SHA-256 kept."""

    def __init__(self) -> None:
        self.docs: dict[str, dict] = {}
        self.sha: dict[str, str] = {}

    def get(self, rel: str) -> dict:
        if rel not in self.docs:
            p = ROOT / rel
            if not p.is_file():
                raise SystemExit(f"{rel} does not exist")
            cm.same_as_head([p])
            raw = p.read_bytes()
            self.docs[rel] = json.loads(raw)
            self.sha[rel] = hashlib.sha256(raw).hexdigest()
        return self.docs[rel]


def drawable(v, where: str) -> str:
    if isinstance(v, bool) or v is None or not isinstance(v, (str, int, float)):
        raise SystemExit(f"{where}: the value {v!r} cannot be drawn as a label")
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            raise SystemExit(f"{where}: the value {v!r} is not finite")
        return f"{v:.3g}"
    if isinstance(v, str) and not v.strip():
        raise SystemExit(f"{where}: the value is an empty string")
    return str(v)


def generator_default(text: str, where: str) -> dict:
    if text not in GENERATOR_DEFAULTS:
        raise SystemExit(f"{where}: no check of the installed generator is fixed in the script for {text!r}")
    cls, field, want = GENERATOR_DEFAULTS[text]
    import atact.field as af

    fields = {f.name: f for f in dataclasses.fields(getattr(af, cls))}
    if field not in fields or fields[field].default != want:
        got = fields[field].default if field in fields else "no such field"
        raise SystemExit(f"{where}: the installed generator has {cls}.{field} = {got!r}, not the default {want!r}")
    return {"class": cls, "field": field, "default": want}


def check_literal(chk: dict, files: Files, where: str) -> dict:
    kinds = [k for k in CHECK_KEYS if k in chk]
    if len(kinds) != 1 or set(chk) != CHECK_KEYS[kinds[0]]:
        raise SystemExit(f"{where}: a check holds exactly the keys of must_contain or of equals, not {sorted(chk)}")
    doc = files.get(chk["file"])
    if "must_contain" in chk:
        keys, needles = chk["keys"], chk["must_contain"]
        if not (isinstance(keys, list) and keys and isinstance(needles, list) and needles
                and all(isinstance(x, str) and x for x in [*keys, *needles])):
            raise SystemExit(f"{where}: must_contain needs non-empty lists of strings")
        texts = []
        for k in keys:
            v = read_key(doc, k, where)
            if not isinstance(v, str):
                raise SystemExit(f"{where}: {chk['file']} {k} is not a text")
            texts.append(v)
        joined = "\n".join(texts)
        absent = [n for n in needles if n not in joined]
        if absent:
            raise SystemExit(f"{where}: {chk['file']} {keys} does not contain {absent}")
        return {"file": chk["file"], "keys": keys, "must_contain": needles, "found": True, "text_checked": joined}
    if "equals" in chk:
        v = read_key(doc, chk["key"], where)
        if type(v) is not type(chk["equals"]) or v != chk["equals"]:
            raise SystemExit(f"{where}: {chk['file']} {chk['key']} is {v!r}, not {chk['equals']!r}")
        return {"file": chk["file"], "key": chk["key"], "equals": chk["equals"], "value": v}


def rule_every_sentence(files: "Files") -> dict:
    n, want = len(files.get(WORK1_RECORD)["sentences"]), files.get("works/conditions/work1.json")["text"]["sentences"]
    if n != want:
        raise SystemExit(f"'every sentence': the record of work 1 holds {n} sentences, the conditions of work 1 name {want}")
    return {"sentences_in_record": n, "text.sentences": want}


def rule_minmax(files: "Files") -> dict:
    if "minmax" not in files.get(WORK1_RECORD)["figures"]:
        raise SystemExit("'per-tile min-max shown next to it': the record of work 1 has no figure minmax")
    return {"figures": sorted(files.get(WORK1_RECORD)["figures"])}


# the literals without a check of their own that the script can still check cheaply (keyed by the where of resolve)
LITERAL_RULES = {"which units (work 1)": rule_every_sentence, "height scale (work 1)": rule_minmax}


def check_made_on(files: "Files") -> dict:
    """The values of each work are tied to the record of the images: the record was made on the conditions read now."""
    out = {}
    for rec, cond in ((WORK1_RECORD, "works/conditions/work1.json"),
                      ("results/work2_sp_color/work2_sp_color.json", "works/conditions/work2_sp_color.json")):
        got = files.get(rec).get("meta", {}).get("conditions_sha256")
        want = cm.sha256(ROOT / cond)
        if got != want:
            raise SystemExit(f"{rec} was not made on {cond} as it is now (meta.conditions_sha256 {got!r}, SHA-256 {want!r})")
        out[rec] = {"conditions": cond, "conditions_sha256": want}
    # the chain of work 2: the record of the main images names the words it was made from by SHA-256, and those words
    # (words.json) were made on the conditions of work2_words.json (band, warp) and work2.json (the lexicon)
    sp = files.get("results/work2_sp_color/work2_sp_color.json")["meta"]
    words_rec = "results/work2_words/words.json"
    words_sha = cm.sha256(ROOT / words_rec)
    files.get(words_rec)
    if sp.get("words_sha256") != words_sha:
        raise SystemExit(f"{words_rec} is not the words of the record of the main images of work 2 "
                         f"(meta.words_sha256 {sp.get('words_sha256')!r}, SHA-256 {words_sha!r})")
    wm = files.docs[words_rec].get("meta", {})
    for key, cond in (("conditions_sha256", "works/conditions/work2_words.json"),
                      ("work2_conditions_sha256", "works/conditions/work2.json")):
        want = cm.sha256(ROOT / cond)
        if wm.get(key) != want:
            raise SystemExit(f"{words_rec} was not made on {cond} as it is now (meta.{key} {wm.get(key)!r}, SHA-256 {want!r})")
        out[f"{words_rec} meta.{key}"] = {"conditions": cond, "sha256": want}
    out["work2_words_record"] = {"file": words_rec, "sha256": words_sha, "named_by": "work2_sp_color record meta.words_sha256"}
    return out


def check_octaves(files: "Files", boxes: dict) -> dict:
    v1 = read_key(files.get("works/conditions/work1.json"), "relief.octaves", "octaves")
    v2 = read_key(files.get("works/conditions/work2_words.json"), "field.octaves", "octaves")
    got = {"work1.json relief.octaves": v1, "work2_words.json field.octaves": v2, "relief.OCTAVES": relief.OCTAVES}
    ok = {k: v for k, v in got.items() if type(v) is int}
    if len(ok) != 3 or len(set(got.values())) != 1 or v1 not in NUMBER_WORDS:
        raise SystemExit(f"the octaves differ or are not one the box names: {got}")
    if f"{NUMBER_WORDS[v1]} octaves" not in boxes["field"]:
        raise SystemExit(f"the box 'field' does not say '{NUMBER_WORDS[v1]} octaves' as the octaves are {v1}")
    return {**got, "box_field_says": f"{NUMBER_WORDS[v1]} octaves"}


def check_warp_absent() -> dict:
    """Work 1 draws with procedural_field_separable, which has no warp argument at all: so none is a fact of the call."""
    import inspect

    import atact.field as af

    if f"{WARP_FUNCTION}(" not in inspect.getsource(work1) or "warp" in inspect.getsource(relief) + inspect.getsource(work1):
        raise SystemExit(f"works/work1.py does not call {WARP_FUNCTION}( or the code of work 1 mentions warp")
    params = list(inspect.signature(getattr(af, WARP_FUNCTION)).parameters)
    if "warp" in params:
        raise SystemExit(f"{WARP_FUNCTION} has a warp parameter ({params}); work 1 would take its default, not none")
    return {"function": WARP_FUNCTION, "parameters": params, "reason": "works/work1.py passes its field through this function (the tile is byte-equal to common.png), and the function has no warp parameter, so work 1's field has none"}


def resolve(spec: dict, files: Files, where: str) -> tuple[str, dict]:
    """The text of one value of a label and how it was obtained."""
    if "file" in spec:
        if set(spec) != SPEC_PATH:
            raise SystemExit(f"{where}: a value read from a file holds exactly file and key, not {sorted(spec)}")
        v = read_key(files.get(spec["file"]), spec["key"], where)
        return drawable(v, where), {"how": "read", "file": spec["file"], "key": spec["key"], "value": v}
    if "literal" not in spec or not set(spec) <= SPEC_LITERAL | SPEC_LITERAL_EXTRA or sum(k in spec for k in ("check", "check_absent")) > 1:
        raise SystemExit(f"{where}: a literal holds literal and at most one of check, check_absent (and a note), not {sorted(spec)}")
    lit = spec["literal"]
    if not isinstance(lit, str) or not lit.strip():
        raise SystemExit(f"{where}: neither a path nor a non-empty literal")
    if "check" in spec and "check_absent" in spec:
        raise SystemExit(f"{where}: both check and check_absent")
    if "check" in spec:
        return lit, {"how": "literal checked", "literal": lit, "check": check_literal(spec["check"], files, where)}
    if "check_absent" in spec:
        ca = spec["check_absent"]
        if set(ca) != ABSENT_KEYS:
            raise SystemExit(f"{where}: check_absent holds exactly file, key and default, not {sorted(ca)}")
        if has_key(files.get(ca["file"]), ca["key"], where):
            raise SystemExit(f"{where}: {ca['file']} has the key {ca['key']!r}; the literal {lit!r} assumes it is absent")
        gen = generator_default(ca["default"], where)
        return lit, {"how": "literal checked absent", "literal": lit,
                     "check_absent": {"file": ca["file"], "key": ca["key"], "absent": True, "default": ca["default"],
                                      "generator": gen}}
    extra = LITERAL_RULES.get(where)
    if extra:
        return lit, {"how": "literal as written, with the check of the script", "literal": lit, "check": extra(files)}
    return lit, {"how": "literal as written", "literal": lit}


# ---------------------------------------------------------------- the tile


def tile_of_work1(files: Files) -> tuple[np.ndarray, dict]:
    """The tile of the first sentence cut from results/work1/common.png at its place, and the same tile drawn afresh by
    works/work1.py; they must be equal byte for byte."""
    rec = files.get(WORK1_RECORD)
    c1 = files.get("works/conditions/work1.json")
    png = ROOT / COMMON_PNG
    cm.same_as_head([png])
    if rec["figures"]["common"]["file"] != "common.png" or cm.sha256(png) != rec["figures"]["common"]["sha256"]:
        raise SystemExit(f"{COMMON_PNG} does not match the SHA-256 of {WORK1_RECORD}")
    s0 = rec["sentences"][0]
    fig, r = c1["figure"], c1["relief"]
    d = work1.dimensions(c1)
    lay = work1.layout(len(rec["sentences"]), fig)
    if lay != rec["layout"] or rec["eps"] != r["eps"]:
        raise SystemExit("the layout or eps of the record of work 1 differs from the conditions of work 1")
    tf = relief.tile(s0["V"], s0["A"], lambda f: procedural_field_separable(
        f, lay["tile_px"], r["seed"], octaves=relief.OCTAVES, normalize=relief.NORMALIZE), r["eps"])
    fresh_fig = work1.figure([relief.grey(tf.H, "common", d)], lay, fig)  # the series' common scale, tile 0 only
    step = (lay["tile_mm"] + fig["gap_mm"]) * fig["dpi"] / MM_PER_INCH
    px = lay["tile_px"]
    y, x = int(round(0 * step)), int(round(0 * step))  # row 0, column 0 of the layout: the first sentence
    with Image.open(png) as im:
        if im.mode != "L":
            raise SystemExit(f"{COMMON_PNG} is {im.mode}, not L")
        cut = np.array(im)[y : y + px, x : x + px]
    fresh = np.array(fresh_fig)[y : y + px, x : x + px]
    if cut.shape != (px, px) or fresh.shape != (px, px):
        raise SystemExit(f"the tile is not {px} x {px} px")
    if not np.array_equal(cut, fresh):
        raise SystemExit(f"the tile cut from {COMMON_PNG} differs from the tile drawn afresh by works/work1.py "
                         f"({int((cut != fresh).sum())} px)")
    return cut, {"source": COMMON_PNG, "source_sha256": rec["figures"]["common"]["sha256"], "sentence_index": 0,
                 "sentence_text": s0["text"], "V": s0["V"], "A": s0["A"], "x_px": x, "y_px": y, "size_px": px,
                 "cut_sha256": hashlib.sha256(cut.tobytes()).hexdigest(),
                 "redrawn_sha256": hashlib.sha256(fresh.tobytes()).hexdigest(), "equal": True}


# ---------------------------------------------------------------- drawing


def check_glyphs(texts: list[str], font_file: Path) -> dict:
    """Every non-ASCII character drawn has a glyph of its own in both faces (a missing one is drawn as the notdef box)."""
    chars = sorted({ch for t in texts for ch in t if ord(ch) > 127})
    out = {}
    for face, idx in FONT_INDEX.items():
        font = ImageFont.truetype(str(font_file), FONT_PX, index=idx)
        notdef = (font.getmask("￿").size, bytes(font.getmask("￿")))
        if (font.getmask("e").size, bytes(font.getmask("e"))) == notdef:
            raise SystemExit("the check of glyphs cannot tell the letter e from the notdef box")
        for ch in chars:
            m = font.getmask(ch)
            if not font.getbbox(ch)[2] or (m.size, bytes(m)) == notdef:
                raise SystemExit(f"the character {ch!r} (U+{ord(ch):04X}) has no glyph in the {face} face of {font_file.name}")
        out[face] = [f"U+{ord(ch):04X}" for ch in chars]
    return out


def wrap(words: list[tuple[str, str]], fonts: dict, draw: ImageDraw.ImageDraw, width: int) -> list[list[tuple[str, str]]]:
    """Lines of at most width px from (word, face) pairs, broken at spaces."""
    lines: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] = []

    def length(ws: list[tuple[str, str]]) -> float:
        return sum(draw.textlength(w, font=fonts[f]) for w, f in ws) + draw.textlength(" ", font=fonts["regular"]) * (len(ws) - 1)

    for w, f in words:
        if draw.textlength(w, font=fonts[f]) > width:
            raise SystemExit(f"{w!r} does not fit in a column {width} px wide")
        if cur and length([*cur, (w, f)]) > width:
            lines.append(cur)
            cur = []
        cur.append((w, f))
    if cur:
        lines.append(cur)
    return lines


def put_lines(draw: ImageDraw.ImageDraw, lines, fonts, x: float, y: float, line_h: int, centre_w: int | None = None) -> None:
    sp = draw.textlength(" ", font=fonts["regular"])
    for ln in lines:
        w = sum(draw.textlength(t, font=fonts[f]) for t, f in ln) + sp * (len(ln) - 1)
        cx = x + ((centre_w - w) / 2 if centre_w else 0)
        for t, f in ln:
            draw.text((cx, y), t, font=fonts[f], fill=0)
            cx += draw.textlength(t, font=fonts[f]) + sp
        y += line_h


def count_words(texts: list[str]) -> int:
    """Whitespace-separated tokens: a path is one word, and so is a mark standing alone (the slash, the dash)."""
    return sum(len(t.split()) for t in texts)


def arrow(draw: ImageDraw.ImageDraw, x0: int, x1: int, y: int) -> None:
    draw.line([(x0, y), (x1 - 8, y)], fill=0, width=3)
    draw.polygon([(x1, y), (x1 - 18, y - 9), (x1 - 18, y + 9)], fill=0)


def main() -> None:
    out_json = OUT_DIR / "rule_schematic.json"
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} exists; refusing to overwrite")
    raw, c = cm.load_conditions(COND)
    cm.same_as_head([COND, *(ROOT / p for p in CODE)])
    state = cm.git_state(["works", "src", "pyproject.toml", "uv.lock", ".python-version"])
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    # the conditions are the ones this script is for, text for text
    seen = {"version": c.get("version"), "top_keys": sorted(c), "layout": c.get("layout"), "boxes": c.get("boxes"),
            "open_quantities": c.get("open_quantities"), "tile": c.get("tile"), "outputs": c.get("outputs")}
    diff = cm.matches_template(APPROVED, seen)
    if diff:
        raise SystemExit("the conditions are not those fixed in the script: " + "; ".join(diff))
    if c["version"] != VERSION:
        raise SystemExit(f"the script is for version {VERSION} of the conditions, not {c['version']}")
    lay = c["layout"]
    dpi = cm.need(lay, "dpi", int, "layout.")
    width = round(cm.need(lay, "width_mm", float, "layout.") * dpi / MM_PER_INCH)
    if lay["colour"] != APPROVED["layout"]["colour"]:
        raise SystemExit("the colour of the layout is not black, white and greys only")
    font_file = Path(lay["font_file"])
    if not font_file.is_file():
        raise SystemExit(f"{font_file} does not exist")
    if FONT_PX * 72 / dpi < MIN_FONT_PT:
        raise SystemExit(f"the font of {FONT_PX} px at {dpi} dpi is below {MIN_FONT_PT} pt")
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    files = Files()
    boxes = {b["id"]: b["text"] for b in c["boxes"]}
    order = [b["id"] for b in c["boxes"]]
    labels: dict[str, list] = {i: [] for i in order}
    for q in c["open_quantities"]["list"]:
        if q["under"] not in labels:
            raise SystemExit(f"{q['name']}: no box {q['under']!r}")
        v1, r1 = resolve(q["work1"], files, f"{q['name']} (work 1)")
        v2, r2 = resolve(q["work2"], files, f"{q['name']} (work 2)")
        labels[q["under"]].append({"name": q["name"], "text": f"{q['name']}:", "work1": v1, "work2": v2,
                                   "record": {"work1": r1, "work2": r2}})
    used = {f"{x['name']} (work {k[-1]})" for i in order for x in labels[i] for k, r in x["record"].items()
            if r["how"] == "literal as written, with the check of the script"}
    if used != set(LITERAL_RULES):
        raise SystemExit(f"the checks of the script for literals {sorted(set(LITERAL_RULES) - used)} were not used")
    octaves = check_octaves(files, boxes)
    warp = check_warp_absent()
    made_on = check_made_on(files)
    tile, tile_rec = tile_of_work1(files)

    # words of the figure: the text of each box and each label as drawn
    drawn_boxes = [boxes[i] for i in order]
    drawn_labels = [f"{x['text']} {x['work1']}{SEP}{x['work2']}" for i in order for x in labels[i]]
    fonts = {k: ImageFont.truetype(str(font_file), FONT_PX, index=i) for k, i in FONT_INDEX.items()}
    faces = {k: list(f.getname()) for k, f in fonts.items()}
    if faces["regular"][1] != "Regular" or faces["bold"][1] != "Bold":
        raise SystemExit(f"the faces of {font_file.name} are not Regular and Bold: {faces}")
    glyphs = check_glyphs([*drawn_boxes, *drawn_labels], font_file)
    words = {"boxes": count_words(drawn_boxes), "labels": count_words(drawn_labels)}
    words["figure"] = words["boxes"] + words["labels"]

    probe = ImageDraw.Draw(Image.new("L", (10, 10), 255))
    margin, gap, pad = 12, 46, 12
    n = len(order)
    col_w = (width - 2 * margin - (n - 1) * gap) // n
    line_h = round(FONT_PX * 1.25)
    if col_w < tile_rec["size_px"] + 2:
        raise SystemExit(f"the tile ({tile_rec['size_px']} px and its frame) does not fit in a column of {col_w} px; "
                         "it is not resampled")
    box_lines = {i: wrap([(w, "regular") for w in boxes[i].split()], fonts, probe, col_w - 2 * pad) for i in order}
    box_h = max(len(v) for v in box_lines.values()) * line_h + 2 * pad
    def label_words(x: dict) -> list[tuple[str, str]]:
        return [(w, "bold") for w in x["text"].split()] + [(w, "regular") for w in f"{x['work1']}{SEP}{x['work2']}".split()]

    label_lines = {i: [wrap(label_words(x), fonts, probe, col_w) for x in labels[i]] for i in order}
    link, between, tile_gap = 26, 16, 14
    tile_px = tile_rec["size_px"]
    col_h = {}
    for i in order:
        h = sum(len(ls) * line_h for ls in label_lines[i]) + between * max(len(label_lines[i]) - 1, 0)
        if i == "image":
            h += tile_px + 2 + tile_gap
        col_h[i] = h
    top = margin
    y_lab = top + box_h + link
    height = y_lab + max(col_h.values()) + margin
    img = Image.new("L", (width, height), 255)
    d = ImageDraw.Draw(img)
    for k, i in enumerate(order):
        x0 = margin + k * (col_w + gap)
        d.rectangle([x0, top, x0 + col_w - 1, top + box_h - 1], fill=236, outline=0, width=3)
        put_lines(d, box_lines[i], fonts, x0 + pad, top + pad + (box_h - 2 * pad - len(box_lines[i]) * line_h) // 2,
                  line_h, centre_w=col_w - 2 * pad)
        if k + 1 < n:
            arrow(d, x0 + col_w + 6, x0 + col_w + gap - 6, top + box_h // 2)
        if labels[i] or i == "image":
            d.line([(x0 + col_w // 2, top + box_h), (x0 + col_w // 2, y_lab - 4)], fill=150, width=2)
        y = y_lab
        if i == "image":
            tx = x0 + (col_w - tile_px) // 2
            d.rectangle([tx - 1, y, tx + tile_px, y + tile_px + 1], outline=0, width=1)
            img.paste(Image.fromarray(tile, mode="L"), (tx, y + 1))
            tile_at = (tx, y + 1)
            y += tile_px + 2 + tile_gap
        for ls in label_lines[i]:
            put_lines(d, ls, fonts, x0, y, line_h)
            y += len(ls) * line_h + between

    OUT_DIR.parent.mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".rule_schematic-", dir=OUT_DIR.parent))
    try:
        img.save(tmp / "rule_schematic.png", dpi=(dpi, dpi))
        with Image.open(tmp / "rule_schematic.png") as saved:
            got_dpi = tuple(round(x) for x in saved.info.get("dpi", (0, 0)))
            if saved.mode != "L" or saved.width != width or got_dpi != (dpi, dpi):
                raise SystemExit(f"the figure is {saved.mode} {saved.size} at {got_dpi} dpi, not L {width} px wide at {dpi}")
            min_pt = FONT_PX * 72 / got_dpi[0]
            if min_pt < MIN_FONT_PT:
                raise SystemExit(f"the smallest type is {min_pt:.2f} pt, below {MIN_FONT_PT} pt")
            shown = np.array(saved)
            pix = set(np.unique(shown).tolist())
            tx, ty = tile_at
            back = shown[ty : ty + tile_px, tx : tx + tile_px]
            if back.shape != tile.shape or not np.array_equal(back, tile):
                raise SystemExit("the tile cut from the saved figure differs from the tile cut from common.png")
            size = saved.size
        res = {"meta": {"conditions_sha256": hashlib.sha256(raw).hexdigest(), "script_sha256": cm.sha256(Path(__file__)),
                        "common_sha256": cm.sha256(ROOT / "works" / "_common.py"), "font_sha256": cm.sha256(font_file),
                        "library_sha256": cm.library_sha256(), **cm.versions("numpy", "pillow"), **state,
                        "started": started, "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
               "files_read_sha256": files.sha,
               "boxes": {i: boxes[i] for i in order},
               "labels": {i: [{"name": x["name"], "drawn": f"{x['text']} {x['work1']}{SEP}{x['work2']}",
                               **x["record"]} for x in labels[i]] for i in order},
               "tile": {**tile_rec, "placed_at_px": list(tile_at), "saved_figure_cut_equals": True,
                        "common_png_sha256_now": cm.sha256(ROOT / COMMON_PNG)},
               "records_made_on_the_conditions": made_on, "octaves": octaves, "warp_of_work1": warp,
               "faces": faces,
               "glyphs_checked": glyphs,
               "words": {**words, "how": "whitespace-separated tokens of the text of each box and each label as drawn, "
                         "the slash and a dash one word each"},
               "drawing": {"width_px": size[0], "height_px": size[1], "width_mm": size[0] * MM_PER_INCH / dpi,
                           "height_mm": size[1] * MM_PER_INCH / dpi, "mode": "L", "dpi": dpi, "font_px": FONT_PX,
                           "min_font_pt": min_pt, "grey_levels": len(pix), "column_px": col_w, "gap_px": gap},
               "outputs": {"rule_schematic.png": cm.sha256(tmp / "rule_schematic.png")}}
        with (tmp / "rule_schematic.json").open("x", encoding="utf-8") as fh:
            fh.write(json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    except BaseException:
        shutil.rmtree(tmp)
        raise
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} appeared during the run; the output is left in {tmp}")
    tmp.rename(OUT_DIR)
    print(json.dumps({"words": words, "drawing": res["drawing"]}, indent=1))


if __name__ == "__main__":
    main()
