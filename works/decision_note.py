"""The template for noting decisions: the quantities a rule leaves open and the decisions changed after the values were
seen, with the reason, the alternatives tried, the images that used each one and who judged the look, filled in for works 1
and 2 and for one decision of the paper, with the same table empty below it, from the conditions and the files they name.

Usage:  uv run python works/decision_note.py        (refuses to overwrite)
Conditions: works/conditions/decision_note.json, version 4 (committed before this version of the script was written)
Input:  the files named in the rows (conditions of works 1 and 2, of the drawing, of the blind judgement and of the record
        of generation, and the images each quantity acted on); every one must be the committed bytes at HEAD
Output: results/decision_note_v4/decision_note.png, decision_note.md and decision_note.json
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
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
OUT_DIR = ROOT / "results" / "decision_note_v4"
CODE = ["works/decision_note.py", "works/_common.py"]
MM_PER_INCH = 25.4
VERSION = 4
ROW_SETS = {"rows_work1": ("Work 1", 2), "rows_work2": ("Work 2", 2), "rows_paper": ("Record of the paper", 1)}
COLUMNS = ["quantity", "value", "reason", "acted_on", "when", "judged"]
ROW_FIELDS = {"quantity", "value", "reason", "acted_on", "compared", "when", "judged", "dagger"}
NONE_RECORDED = "none recorded"
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
    if set(v) in ({"literal"}, {"literal", "check"}):
        text = v["literal"]
        if not isinstance(text, str) or not text:
            raise SystemExit(f"{where}: the literal is not a non-empty string")
        if "check" not in v:
            return text, {"literal": True}
        ch = v["check"]
        if not isinstance(ch, dict) or set(ch) != {"file", "keys", "must_contain"}:
            raise SystemExit(f"{where}: the check needs a file, keys and must_contain")
        must = strings(ch["must_contain"], where, "must_contain")
        c = load_json(ch["file"], cache)
        read = {k: as_text(read_key(c, k, where), where) for k in strings(ch["keys"], where, "keys")}
        missing = [s for s in must if not any(s in x for x in read.values())]
        if missing:
            raise SystemExit(f"{where}: not found in {ch['keys']} of {ch['file']}: {missing}")
        return text, {"literal": True, "check": {"file": ch["file"], "read": read, "found": ch["must_contain"]}}
    raise SystemExit(f"{where}: the value is neither a file with keys and format nor a literal")


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
    if not isinstance(r, dict) or set(r) != {"why", "tried"}:
        raise SystemExit(f"{where}.reason needs why and tried")
    if not isinstance(r["why"], str) or not r["why"]:
        raise SystemExit(f"{where}.reason.why is not a non-empty string")
    if not isinstance(r["tried"], list) or not all(isinstance(t, str) and t for t in r["tried"]):
        raise SystemExit(f"{where}.reason.tried is not a list of non-empty strings")
    j = row["judged"]
    if isinstance(j, str):
        if j != NONE_RECORDED:
            raise SystemExit(f"{where}.judged: a string must be {NONE_RECORDED!r}; anything else is a literal")
    elif not isinstance(j, dict) or "literal" not in j:
        raise SystemExit(f"{where}.judged is neither {NONE_RECORDED!r} nor a literal")


def reason_text(r: dict) -> str:
    tried = "; ".join(r["tried"]) if r["tried"] else "none"
    return f"{r['why'][:1].upper()}{r['why'][1:]}. Tried: {tried}."


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
    mark_d, meaning_d = cm.need(dag, "mark", str, "dagger."), cm.need(dag, "meaning", str, "dagger.")
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
                ch = v.get("check", {})
                for f in (v.get("file"), ch.get("file") if isinstance(ch, dict) else None):
                    if f is not None:
                        if not isinstance(f, str) or not (ROOT / f).is_file():
                            raise SystemExit(f"{where}.{what}: {f} does not exist")
                        named.add(f)
            named.update(row["acted_on"])
            if not isinstance(row["value"], dict):
                raise SystemExit(f"{where}.value is not an object")
            text, read = resolve_value(row["value"], cache, where + ".value")
            if isinstance(row["judged"], dict):
                if set(row["judged"]) not in ({"literal"}, {"literal", "check"}):
                    raise SystemExit(f"{where}.judged is not a literal")
                judged, judged_read = resolve_value(row["judged"], cache, where + ".judged")
            else:
                judged, judged_read = row["judged"], {"literal": True}
            resolved[key].append({
                "quantity": row["quantity"], "dagger": row["dagger"], "value": text, "reason": reason_text(row["reason"]),
                "acted_on": [p + (mark_c if p in row["compared"] else "") for p in row["acted_on"]],
                "when": row["when"], "judged": judged, "read": {"value": read, "judged": judged_read}})
    if not font_file.is_file():
        raise SystemExit(f"{font_file} does not exist")
    daggered = [r["quantity"] for rs in resolved.values() for r in rs if r["dagger"]]

    files = sorted(named)
    cm.same_as_head([COND, *(ROOT / p for p in CODE), *(ROOT / p for p in files)])
    state = cm.git_state(["works", *files, "pyproject.toml", "uv.lock"])
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    def cells(r: dict) -> list[str]:
        q = (mark_d if r["dagger"] else "") + r["quantity"]
        return [q, r["value"], r["reason"], r["acted_on"], r["when"], r["judged"]]

    filled_rows: list = []
    for key, (band, _) in ROW_SETS.items():
        filled_rows.append(band)
        filled_rows += [[[x] if isinstance(x, str) else x for x in cells(r)] for r in resolved[key]]
    legend = (mark_d + meaning_d) if daggered else ""

    fonts = {k: ImageFont.truetype(str(font_file), size, index=i) for k, i in FONT_INDEX.items()}
    shares = [share[k] for k in COLUMNS]
    parts = [title_image(titles[0], width, fonts, dpi), table(headers, filled_rows, shares, width, fonts, dpi)]
    if legend:
        parts += [Image.new("L", (width, round(TITLE_GAP_PT * dpi / 72)), 255),
                  title_image(legend, width, {"bold": fonts["regular"]}, dpi, one_line=True)]
    gap = Image.new("L", (width, round(TABLE_GAP_PT * dpi / 72)), 255)
    parts += [gap, title_image(titles[1], width, fonts, dpi),
              table(headers, [[[""]] * len(headers)] * EMPTY_ROWS, shares, width, fonts, dpi, min_lines=EMPTY_ROW_LINES)]
    fig = Image.new("L", (width, sum(p.height for p in parts)), 255)
    y = 0
    for p in parts:
        fig.paste(p, (0, y))
        y += p.height
    drawn_filled = [titles[0], *headers, *(b for b, _ in ROW_SETS.values()), legend,
                    *(t for rs in resolved.values() for r in rs for x in cells(r) for t in ([x] if isinstance(x, str) else x))]
    words = {"filled": count_words(drawn_filled), "empty": count_words([titles[1], *headers])}
    words["total"] = words["filled"] + words["empty"]
    if words["total"] > max_words:
        raise SystemExit(f"the two tables have {words['total']} words, more than {max_words}")

    md_rows: list = []
    for key, (band, _) in ROW_SETS.items():
        md_rows.append(band)
        md_rows += [[x if isinstance(x, str) else ", ".join(x) for x in cells(r)] for r in resolved[key]]
    md = (f"# {titles[0]}\n\n" + md_table(headers, md_rows) + (f"\n{legend}\n" if legend else "")
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
                        **cm.versions("pillow"), **state, "started": started,
                        "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
               "files_named_sha256": {p: cm.sha256(ROOT / p) for p in files},
               **resolved,
               "dagger_rows": daggered,
               "words": {**words, "max_words": max_words, "how": "whitespace-separated tokens, a path one word, "
                         "a mark standing alone one word; titles, headers, band labels, the legend and the cells"},
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
