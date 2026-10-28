"""The template for noting decisions: the quantities a rule leaves open and the decisions changed after the values were
seen, filled in for work 1, with the same table empty next to it, from the conditions and the files they name.

Usage:  uv run python works/decision_note.py        (refuses to overwrite)
Conditions: works/conditions/decision_note.json (committed before this script was written)
Input:  the files named in the rows (conditions of work 1 and of its drawing, and the images each quantity acted on);
        every one must be the committed bytes at HEAD
Output: results/decision_note/decision_note.png, decision_note.md and decision_note.json
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
OUT_DIR = ROOT / "results" / "decision_note"
CODE = ["works/decision_note.py", "works/_common.py"]
MM_PER_INCH = 25.4
# Drawing choices the conditions leave open (decision 1 of .steering/20261028-decision-note)
FONT_PT = 8.0
FONT_INDEX = {"regular": 0, "bold": 1}  # faces of Helvetica.ttc
COLUMN_SHARE = {"quantity": 0.15, "value": 0.29, "acted_on": 0.30, "when": 0.26}
PAD_PT = 3.0
RULE_PX = 2
TABLE_GAP_PT = 24.0
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


def table(headers: list[str], rows: list[list[list[str]]], width: int, fonts: dict, dpi: int,
          min_lines: int = 1) -> Image.Image:
    """A ruled table: each cell is a list of paragraphs (one per line of the record, e.g. one per image)."""
    pad = round(PAD_PT * dpi / 72)
    line_h = round(FONT_PT * 1.3 * dpi / 72)
    widths = [round(COLUMN_SHARE[k] * width) for k in COLUMN_SHARE]
    widths[-1] = width - sum(widths[:-1])
    scratch = ImageDraw.Draw(Image.new("L", (1, 1)))

    def lay(cells: list[list[str]], font) -> list[list[str]]:
        return [[ln for par in cell for ln in wrap(scratch, par, font, w - 2 * pad)] for cell, w in zip(cells, widths)]

    laid = [lay([[h] for h in headers], fonts["bold"])] + [lay(r, fonts["regular"]) for r in rows]
    heights = [max(min_lines if i else 1, *(len(c) for c in r)) * line_h + 2 * pad for i, r in enumerate(laid)]
    img = Image.new("L", (width, sum(heights) + RULE_PX), 255)
    draw = ImageDraw.Draw(img)
    y = 0
    for i, (r, h) in enumerate(zip(laid, heights)):
        x = 0
        for cell, w in zip(r, widths):
            for j, ln in enumerate(cell):
                draw.text((x + pad, y + pad + j * line_h), ln, fill=0, font=fonts["bold" if i == 0 else "regular"])
            x += w
        draw.rectangle((0, y, width - 1, y + RULE_PX - 1), fill=0)
        y += h
    draw.rectangle((0, y, width - 1, y + RULE_PX - 1), fill=0)
    x = 0
    for w in [0, *widths]:
        x += w
        draw.rectangle((min(x, width - RULE_PX), 0, min(x, width - RULE_PX) + RULE_PX - 1, img.height - 1), fill=0)
    return img


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    def cell(s: str) -> str:
        return s.replace("|", "\\|")

    out = ["| " + " | ".join(cell(h) for h in headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(cell(x) for x in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def main() -> None:
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} exists; refusing to overwrite")
    raw, c = cm.load_conditions(COND)
    if cm.need(c, "version", int) != 2:
        raise SystemExit(f"the script is for version 2 of the conditions, not {c['version']}")
    cols = cm.need(c, "columns", list)
    ids = [col["id"] for col in cols]
    if ids != list(COLUMN_SHARE):
        raise SystemExit(f"the columns {ids} are not those the script draws {list(COLUMN_SHARE)}")
    headers = [cm.need(col, "header", str, "columns.") for col in cols]
    rows = cm.need(c, "rows_work1", list)
    lay = cm.need(c, "layout", dict)
    if lay.get("colour") != "black and white":
        raise SystemExit(f"layout.colour is {lay.get('colour')!r}; the script draws in black and white only")
    font_file = Path(cm.need(lay, "font_file", str, "layout."))
    width = round(cm.need(lay, "width_mm", float, "layout.") * cm.need(lay, "dpi", int, "layout.") / MM_PER_INCH)
    dpi = lay["dpi"]
    label = cm.need(c, "acted_on_labels", str)

    cache: dict = {}
    named: set[str] = set()
    resolved = []
    for i, row in enumerate(rows):
        where = f"rows_work1[{i}]"
        if set(row) != set(ids):
            raise SystemExit(f"{where}: fields {sorted(row)} are not the columns {ids}")
        for k in ("quantity", "when"):
            if not isinstance(row[k], str) or not row[k]:
                raise SystemExit(f"{where}.{k} is not a non-empty string")
        acted = row["acted_on"]
        if not isinstance(acted, list) or not acted or not all(isinstance(p, str) and p for p in acted):
            raise SystemExit(f"{where}.acted_on is not a non-empty list of paths")
        if len(set(acted)) != len(acted):
            raise SystemExit(f"{where}.acted_on names an image twice")
        for p in acted:
            if not (ROOT / p).is_file():
                raise SystemExit(f"{where}.acted_on: {p} does not exist")
        v = row["value"]
        if not isinstance(v, dict):
            raise SystemExit(f"{where}.value is not an object")
        ch = v.get("check", {})
        for f in (v.get("file"), ch.get("file") if isinstance(ch, dict) else None):
            if f is not None:
                if not isinstance(f, str) or not (ROOT / f).is_file():
                    raise SystemExit(f"{where}.value: {f} does not exist")
                named.add(f)
        named.update(acted)
        text, read = resolve_value(v, cache, where)
        resolved.append({"quantity": row["quantity"], "value": text, "acted_on": acted, "when": row["when"],
                         "read": read})
    if not font_file.is_file():
        raise SystemExit(f"{font_file} does not exist")

    files = sorted(named)
    cm.same_as_head([COND, *(ROOT / p for p in CODE), *(ROOT / p for p in files)])
    state = cm.git_state(["works", *files, "pyproject.toml", "uv.lock"])
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    size = round(FONT_PT * dpi / 72)
    fonts = {k: ImageFont.truetype(str(font_file), size, index=i) for k, i in FONT_INDEX.items()}
    filled = table(headers, [[[r["quantity"]], [r["value"]], r["acted_on"], [r["when"]]] for r in resolved],
                   width, fonts, dpi)
    empty = table(headers, [[[""]] * len(headers)] * EMPTY_ROWS, width, fonts, dpi, min_lines=EMPTY_ROW_LINES)
    gap = round(TABLE_GAP_PT * dpi / 72)
    fig = Image.new("L", (width, filled.height + gap + empty.height), 255)
    fig.paste(filled, (0, 0))
    fig.paste(empty, (0, filled.height + gap))

    md_rows = [[r["quantity"], r["value"], ", ".join(r["acted_on"]), r["when"]] for r in resolved]
    md = ("# Decision note, filled in for work 1\n\n" + md_table(headers, md_rows)
          + "\n" + label[0].upper() + label[1:] + ".\n"
          + "\n# Decision note, empty\n\n" + md_table(headers, [[""] * len(headers)] * EMPTY_ROWS))

    OUT_DIR.parent.mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".decision_note-", dir=OUT_DIR.parent))
    try:
        fig.save(tmp / "decision_note.png", dpi=(dpi, dpi))
        (tmp / "decision_note.md").write_text(md, encoding="utf-8")
        res = {"meta": {"conditions_sha256": hashlib.sha256(raw).hexdigest(),
                        "script_sha256": cm.sha256(Path(__file__)),
                        "common_sha256": cm.sha256(ROOT / "works" / "_common.py"), "font_sha256": cm.sha256(font_file),
                        **cm.versions("pillow"), **state, "started": started,
                        "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
               "files_named_sha256": {p: cm.sha256(ROOT / p) for p in files},
               "rows_work1": resolved,
               "drawing": {"width_px": width, "dpi": dpi, "font_pt": FONT_PT, "column_share": COLUMN_SHARE,
                           "empty_rows": EMPTY_ROWS, "empty_row_lines": EMPTY_ROW_LINES, "minor_words": sorted(MINOR_WORDS)},
               "outputs": {n: cm.sha256(tmp / n) for n in ("decision_note.png", "decision_note.md")}}
        with (tmp / "decision_note.json").open("x", encoding="utf-8") as fh:
            fh.write(json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    except BaseException:
        shutil.rmtree(tmp)
        raise
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} appeared during the run; the output is left in {tmp}")
    tmp.rename(OUT_DIR)
    print(json.dumps({r["quantity"]: r["value"] for r in resolved}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
