"""The template for noting decisions: the quantities a rule leaves open and the decisions changed after the values were
seen, with the reason, the alternatives tried, the images that used each one and who judged the look, filled in for works 1
and 2, with the same table empty below it, from the conditions and the files they name.

Usage:  uv run python works/decision_note.py        (refuses to overwrite)
Conditions: works/conditions/decision_note.json, version 6 (committed before this version of the script was written)
Input:  the files named in the rows (conditions of works 1 and 2, of the drawing, of the blind judgement and of the record
        of generation, the scoring of the blind judgement and its supplement, and the images each quantity acted on); every
        one must be the committed bytes at HEAD. The range of f and s is read from the installed atact.mapping, which must be
        the library of the record of generation
Output: results/decision_note_v6/decision_note.png, decision_note.md and decision_note.json
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
OUT_DIR = ROOT / "results" / "decision_note_v6"
CODE = ["works/decision_note.py", "works/_common.py"]
MM_PER_INCH = 25.4
VERSION = 6
ROW_SETS = {"rows_work1": ("Work 1", 2), "rows_work2": ("Work 2", 2)}
COLUMNS = ["quantity", "value", "reason", "acted_on", "when", "judged"]
ROW_FIELDS = {"quantity", "value", "reason", "acted_on", "compared", "when", "judged", "dagger"}
NONE_RECORDED = "none recorded"
# answers_rule of the conditions: the phrase of each outcome and of each row is fixed here, not in the rows
OUTCOME_PHRASE = {"both differ": "both said the rows differ", "neither": "neither said the rows differ", "split": "they split"}
ROW_PHRASE = {"words20": "the 20 words"}
# the version each quantity of work 2 speaks of (G0: map B, P: the percentiles of blind_judgement.json)
VERSION_OF = {"lexicon and map": "G0", "percentiles": "P"}
# a text before a built phrase may not make a claim of its own: no clause of its own and none of these words
SAID = re.compile(r";|\b(?:rows?|words|range|ranges|agreed?|disagreed?|differ|differs|split|both|neither|all|G0|P|SPC)\b")
JUDGEMENT = "works/conditions/blind_judgement.json"  # its rows are the rows an answer may name
SHARE = re.compile(r"(\d+)%")
USED_QUANTITIES = ["f", "s"]
# used_range_rule of the conditions: the phrase of the share is fixed here
USED_PHRASE = "{text}; f and s of the {words} words use {f} and {s} of their ranges (min to max; counted after the record of generation was seen)"
PERCENT = re.compile(r"(?<![\d.])(?:over )?\d+(?:\.\d+)?%")


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
TABLE_GAP_PT = 24.0
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


def resolve_value(v: dict, cache: dict, where: str) -> tuple[str, dict]:
    """The text of a value and what was read for it."""
    if set(v) == {"file", "keys", "format"}:
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
        return text, {"file": v["file"], "read": read}
    if set(v) in ({"literal"}, {"literal", "check"}, {"text", "check", "used_range"}, {"text", "check", "answers"}):
        base = v["literal"] if "literal" in v else v["text"]
        if not isinstance(base, str) or not base:
            raise SystemExit(f"{where}: the literal or text is not a non-empty string")
        if "literal" in v and PERCENT.search(base):
            raise SystemExit(f"{where}: a percentage in a literal (required_checks)")
        extra, text = {}, base
        if "used_range" in v:
            extra["used_range"], text = check_used_range(v["used_range"], base, cache, where)
        if "answers" in v:
            extra["answers"], text = check_answers(v["answers"], base, cache, where)
        if any(mk in base for mk in ("†", "*")) or PERCENT.search(base):
            raise SystemExit(f"{where}: a reserved mark or a percentage in the text {base!r}")
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


def check_used_range(u: dict, base: str, cache: dict, where: str) -> tuple[dict, str]:
    """The share of the range of f and s used by the words of the record, checked against the strings shown, and the value
    built from it (used_range_rule)."""
    if not isinstance(u, dict) or set(u) != {"file", "version", "quantities", "words_key", "words", "shown"}:
        raise SystemExit(f"{where}: used_range needs file, version, quantities, words_key, words and shown")
    rec = load_json(u["file"], cache)
    meta = read_key(rec, "meta", where)
    if not isinstance(meta, dict):
        raise SystemExit(f"{where}: the meta of {u['file']} is not an object")
    if cm.installed_commit() != meta.get("atact_commit") or cm.library_sha256() != meta.get("library_sha256"):
        raise SystemExit(f"{where}: the installed atact is not the library of {u['file']}")
    per = read_key(rec, u["words_key"], where)
    n = u["words"]
    if (isinstance(n, bool) or not isinstance(n, int) or not isinstance(per, list)
            or not all(isinstance(x, int) and not isinstance(x, bool) for x in per) or sum(per) != n):
        raise SystemExit(f"{where}: the words per author {per!r} of the record do not sum to {n!r}")
    listed = read_key(rec, f"words.{u['version']}", where) if isinstance(u["version"], str) else None
    if not isinstance(listed, dict) or sum(len(x) for x in listed.values() if isinstance(x, list)) != n:
        raise SystemExit(f"{where}: the record does not list {n!r} words under {u['version']!r}")
    if not isinstance(u["version"], str) or not u["version"]:
        raise SystemExit(f"{where}: used_range.version is not a non-empty string")
    if strings(u["quantities"], where, "quantities") != USED_QUANTITIES:
        raise SystemExit(f"{where}: the quantities are not {USED_QUANTITIES}")
    shown = u["shown"]
    if not isinstance(shown, dict) or list(shown) != USED_QUANTITIES:
        raise SystemExit(f"{where}: shown is not one string for each of {USED_QUANTITIES}")
    ranges = rule_ranges()
    shares = {}
    for q in USED_QUANTITIES:
        m = SHARE.fullmatch(shown[q]) if isinstance(shown[q], str) else None
        if m is None:
            raise SystemExit(f"{where}: shown.{q} {shown[q]!r} is not 'N%'")
        if q not in ranges:
            raise SystemExit(f"{where}: no range of the rule is known for {q}")
        lo, hi = ranges[q]
        pooled = read_key(rec, f"pooled.{u['version']}.{q}", where)
        if not isinstance(pooled, dict):
            raise SystemExit(f"{where}: pooled.{u['version']}.{q} is not an object")
        top, bottom = (number(read_key(pooled, k, where), where) for k in ("max", "min"))
        shares[q] = 100 * (top - bottom) / (hi - lo)
        if round(shares[q]) != int(m.group(1)):
            raise SystemExit(f"{where}: {q} uses {shares[q]:.2f}% of its range, which is not {shown[q]}")
    text = USED_PHRASE.format(text=base, words=n, f=f"{round(shares['f'])}%", s=f"{round(shares['s'])}%")
    return {"file": u["file"], "version": u["version"], "ranges": ranges, "shares_percent": shares, "shown": shown}, text


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
    for k in ("quantity", "when"):
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
    if not isinstance(r, dict) or set(r) != {"why", "tried", "not_tried"}:
        raise SystemExit(f"{where}.reason needs why, tried and not_tried")
    if not isinstance(r["why"], str) or not r["why"]:
        raise SystemExit(f"{where}.reason.why is not a non-empty string")
    for k in ("tried", "not_tried"):
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
    not_tried = f" Not tried: {'; '.join(r['not_tried'])}." if r["not_tried"] else ""
    return f"{r['why'][:1].upper()}{r['why'][1:]}. Tried: {tried}.{not_tried}"


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
    note = cm.need(c, "note", str)
    if not note or any(mk.strip() in note for mk in (mark_c, mark_d)):
        raise SystemExit("the note is empty or holds a reserved mark")
    max_words = cm.need(c, "max_words", int)
    size = round(FONT_PT * dpi / 72)
    if size * 72 / dpi < MIN_FONT_PT:
        raise SystemExit(f"the font of {size} px at {dpi} dpi is below {MIN_FONT_PT} pt")

    cache: dict = {}
    named: set[str] = set()
    resolved: dict[str, list] = {}
    for key, (band, n) in ROW_SETS.items():
        rows = cm.need(c, key, list)
        if len(rows) != n:
            raise SystemExit(f"{key} has {len(rows)} rows, not {n}")
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
            text, read = resolve_value(row["value"], cache, where + ".value")
            if isinstance(row["judged"], dict):
                if set(row["judged"]) not in ({"literal"}, {"literal", "check"}, {"text", "check", "answers"}):
                    raise SystemExit(f"{where}.judged is neither a literal nor a text with answers")
                judged, judged_read = resolve_value(row["judged"], cache, where + ".judged")
            else:
                judged, judged_read = row["judged"], {"literal": True}
            reserved = [mk.strip() for mk in (mark_c, mark_d)]
            plain = [row["quantity"], text, reason_text(row["reason"]), *row["acted_on"], row["when"], judged]
            bad = [t for t in plain if any(mk in t for mk in reserved)]
            if bad:
                raise SystemExit(f"{where}: a reserved mark {reserved} is in {bad}")
            j = row["judged"]
            if isinstance(j, dict) and "literal" in j and SAID.search(j["literal"]):
                raise SystemExit(f"{where}.judged: a literal that says an outcome or names words (required_checks)")
            u = row["value"].get("used_range")
            if key == "rows_work2" and (not isinstance(u, dict) or VERSION_OF.get(row["quantity"]) != u.get("version")):
                raise SystemExit(f"{where}: the version of {row['quantity']!r} is {VERSION_OF.get(row['quantity'])!r}, "
                                 f"not that of its used_range")
            if isinstance(j, dict) and "answers" in j:
                u = row["value"].get("used_range")
                if not isinstance(u, dict) or u.get("version") != j["answers"].get("version"):
                    raise SystemExit(f"{where}: the version of the answers is not the version of the used_range of the row")
            resolved[key].append({
                "quantity": row["quantity"], "dagger": row["dagger"], "value": text, "reason": reason_text(row["reason"]),
                "acted_on": [p + (mark_c if p in row["compared"] else "") for p in row["acted_on"]],
                "when": row["when"], "judged": judged, "read": {"value": read, "judged": judged_read}})
    if not font_file.is_file():
        raise SystemExit(f"{font_file} does not exist")
    daggered = [r["quantity"] for rs in resolved.values() for r in rs if r["dagger"]]
    if daggered:
        raise SystemExit(f"rows {daggered} carry the dagger; version 6 has none in the main table")

    files = sorted(named)
    cm.same_as_head([COND, *(ROOT / p for p in CODE), *(ROOT / p for p in files)])
    state = cm.git_state(["works", *files, "pyproject.toml", "uv.lock"])
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    def cells(r: dict) -> list[str]:
        return [r["quantity"], r["value"], r["reason"], r["acted_on"], r["when"], r["judged"]]

    filled_rows: list = []
    for key, (band, _) in ROW_SETS.items():
        filled_rows.append(band)
        filled_rows += [[[x] if isinstance(x, str) else x for x in cells(r)] for r in resolved[key]]

    fonts = {k: ImageFont.truetype(str(font_file), size, index=i) for k, i in FONT_INDEX.items()}
    shares = [share[k] for k in COLUMNS]
    parts = [title_image(titles[0], width, fonts, dpi), table(headers, filled_rows, shares, width, fonts, dpi)]
    parts += [Image.new("L", (width, round(TITLE_GAP_PT * dpi / 72)), 255),
              title_image(note, width, {"bold": fonts["regular"]}, dpi, one_line=True)]
    gap = Image.new("L", (width, round(TABLE_GAP_PT * dpi / 72)), 255)
    parts += [gap, title_image(titles[1], width, fonts, dpi),
              table(headers, [[[""]] * len(headers)] * EMPTY_ROWS, shares, width, fonts, dpi, min_lines=EMPTY_ROW_LINES)]
    fig = Image.new("L", (width, sum(p.height for p in parts)), 255)
    y = 0
    for p in parts:
        fig.paste(p, (0, y))
        y += p.height
    drawn_filled = [titles[0], *headers, *(b for b, _ in ROW_SETS.values()), note,
                    *(t for rs in resolved.values() for r in rs for x in cells(r) for t in ([x] if isinstance(x, str) else x))]
    words = {"filled": count_words(drawn_filled), "empty": count_words([titles[1], *headers])}
    words["total"] = words["filled"] + words["empty"]
    if words["total"] > max_words:
        raise SystemExit(f"the two tables have {words['total']} words, more than {max_words}")

    md_rows: list = []
    for key, (band, _) in ROW_SETS.items():
        md_rows.append(band)
        md_rows += [[x if isinstance(x, str) else ", ".join(x) for x in cells(r)] for r in resolved[key]]
    md = (f"# {titles[0]}\n\n" + md_table(headers, md_rows) + f"\n{note}\n"
          + "\n" + label[0].upper() + label[1:] + ".\n"
          + f"\n# {titles[1]}\n\n" + md_table(headers, [[""] * len(headers)] * EMPTY_ROWS))

    OUT_DIR.parent.mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".decision_note-", dir=OUT_DIR.parent))
    try:
        fig.save(tmp / "decision_note.png", dpi=(dpi, dpi))
        with Image.open(tmp / "decision_note.png") as saved:
            got_dpi = tuple(round(d) for d in saved.info.get("dpi", (0, 0)))
            if saved.mode != "L" or saved.width != width or got_dpi != (dpi, dpi):
                raise SystemExit(f"the saved image is {saved.mode} {saved.size} at {got_dpi} dpi, not L {width} px wide at {dpi}")
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
               "words": {**words, "max_words": max_words, "how": "whitespace-separated tokens, a path one word, "
                         "a mark standing alone one word; titles, headers, band labels, the note and the cells"},
               "drawing": {"width_px": width, "height_px": fig.height, "mode": fig.mode, "dpi": dpi, "font_pt": FONT_PT,
                           "column_share": share, "empty_rows": EMPTY_ROWS, "empty_row_lines": EMPTY_ROW_LINES,
                           "minor_words": sorted(MINOR_WORDS)},
               "outputs": {n: cm.sha256(tmp / n) for n in ("decision_note.png", "decision_note.md")}}
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
