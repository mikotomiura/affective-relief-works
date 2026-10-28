"""Work 2, the blind judgement: rows of tiles of the two trial authors under five common seeds in three versions (grey as
before, grey from percentiles, relief from percentiles with a seed per spelling and colour), shown as pairs in random
order, the authors not named, in a spreadsheet for a judge who has not seen the sheets; the key sealed outside git.

Usage:  uv run --with openpyxl==3.1.5 python works/blind_judgement.py        (refuses to overwrite)
Conditions: works/conditions/blind_judgement.json (committed before this script was run)
Input:  results/sketch_sp_color/values.json (percentiles), results/work2_words/words.json,
        results/work2_variants/variants.json; the drawing of works/sketch_sp_color.py
Output: advice/blind-judgement/judgement.xlsx and advice/blind-judgement/sealed/key.json (outside git),
        results/blind_judgement/record.json (the hashes and checks, nothing that reveals the key)
"""

from __future__ import annotations

import datetime as dt
import hashlib
import html
import io
import json
import random
import re
import secrets
import shutil
import struct
import sys
import tempfile
import zipfile
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.pycache_prefix = tempfile.mkdtemp(prefix="pycache-")  # never run a stale .pyc (ignored by git)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "works"))
sys.path.insert(0, str(ROOT / "checks"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import _common as cm  # noqa: E402
import relief  # noqa: E402
from sketch_color_features import grey_u8, spelling_seed  # noqa: E402
from sketch_sp_color import chroma_of, hue_of, oklch_tint  # noqa: E402
from work2_words import z_mm  # noqa: E402

COND = ROOT / "works" / "conditions" / "blind_judgement.json"
SKETCH_COND = ROOT / "works" / "conditions" / "sketch_sp_color.json"
WORDS_COND = ROOT / "works" / "conditions" / "work2_words.json"
CODE = ["works/blind_judgement.py", "works/sketch_sp_color.py", "works/sketch_color_features.py", "works/work2_words.py",
        "works/_common.py", "src/relief.py"]
OUT_DIR = ROOT / "advice" / "blind-judgement"
REC_DIR = ROOT / "results" / "blind_judgement"
KINDS = ("test", "same_image", "seed_differs", "set_differs", "repeat")
MIN_REPEAT_DISTANCE = 10
GREY = ("G0", "P")  # the versions of the seed_differs controls (in SPC the colours do not change with the seed)
ROW_GAP = 24
WHITE = 255


# ---------------------------------------------------------------- tiles and rows


def row_image(tiles: list[np.ndarray], gap: int) -> np.ndarray:
    """RGB tiles side by side with white gaps."""
    px = tiles[0].shape[0]
    out = np.full((px, len(tiles) * px + (len(tiles) - 1) * gap, 3), WHITE, dtype=np.uint8)
    for k, t in enumerate(tiles):
        out[:, k * (px + gap) : k * (px + gap) + px] = t
    return out


def sketch_tile_place(row_index: int, line: int, k: int, px: int, lines_per_block: int = 4) -> tuple[int, int]:
    """Top left (y, x) of a tile in the figures of works/sketch_color_features.py figure()."""
    gap, lab, bgap, title = 3, 150, 18, 22
    y = row_index * (title + lines_per_block * (px + gap) + bgap) + title + line * (px + gap)
    return y, lab + k * (px + gap)


def canvas_size(t: dict) -> tuple[int, int]:
    w20 = 20 * t["words20_px"] + 19 * t["gap"]
    w10 = 10 * t["sets_px"] + 9 * t["gap"]
    return max(w20, w10), 2 * max(t["words20_px"], t["sets_px"]) + ROW_GAP


def compose(top: np.ndarray, bottom: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Two rows on a white canvas of the given size, each at the left of its half and centred in height."""
    W, H = size
    half = (H - ROW_GAP) // 2
    img = np.full((H, W, 3), WHITE, dtype=np.uint8)
    for r, y0 in ((top, 0), (bottom, half + ROW_GAP)):
        if r.shape[0] > half or r.shape[1] > W:
            raise ValueError(f"a row of {r.shape} does not fit a canvas of {size}")
        y = y0 + (half - r.shape[0]) // 2
        img[y : y + r.shape[0], : r.shape[1]] = r
    return img


def png_bytes(a: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(a, mode="RGB" if a.ndim == 3 else "L").save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def png_chunks(b: bytes) -> list[str]:
    if b[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    out, i = [], 8
    while i < len(b):
        (n,) = struct.unpack(">I", b[i : i + 4])
        out.append(b[i + 4 : i + 8].decode("ascii"))
        i += 12 + n
    return out


# ---------------------------------------------------------------- draws


def draw_items(rng: random.Random, versions: list[str], rows: list[str], sets: list[str], seeds: list[int]) -> list[dict]:
    """The items in the order shown. A side is (version, row, seed, author index)."""
    def side(v: str, row: str, seed: int, author: int) -> dict:
        return {"version": v, "row": row, "seed": seed, "author": author}

    tests = [{"kind": "test", "pair": [side(v, r, s, 0), side(v, r, s, 1)]} for v in versions for r in rows for s in seeds]
    same = []
    for r in rows:
        x = side(rng.choice(versions), r, rng.choice(seeds), rng.randrange(2))
        same.append({"kind": "same_image", "pair": [x, dict(x)]})
    seed_diff = []
    for r in rows:
        v, a, (s1, s2) = rng.choice(GREY), rng.randrange(2), rng.sample(seeds, 2)
        seed_diff.append({"kind": "seed_differs", "pair": [side(v, r, s1, a), side(v, r, s2, a)]})
    set_diff = []
    for p, q in rng.sample(list(combinations(sets, 2)), 5):
        v, a, s = rng.choice(versions), rng.randrange(2), rng.choice(seeds)
        set_diff.append({"kind": "set_differs", "pair": [side(v, p, s, a), side(v, q, s, a)]})
    for it in tests + same + seed_diff + set_diff:
        if rng.randrange(2):
            it["pair"].reverse()
    repeats = []
    for r in rows:
        v, s = rng.choice(versions), rng.choice(seeds)
        first = next(t for t in tests if (t["pair"][0]["version"], t["pair"][0]["row"], t["pair"][0]["seed"]) == (v, r, s))
        repeats.append({"kind": "repeat", "pair": [dict(first["pair"][1]), dict(first["pair"][0])], "of": first})
    items = tests + same + seed_diff + set_diff + repeats
    while True:
        rng.shuffle(items)
        pos = {id(it): n for n, it in enumerate(items)}
        if all(pos[id(it)] - pos[id(it["of"])] >= MIN_REPEAT_DISTANCE for it in repeats):
            break
    pos = {id(it): n for n, it in enumerate(items)}
    out = []
    for it in items:
        o = {"kind": it["kind"], "top": it["pair"][0], "bottom": it["pair"][1]}
        if it["kind"] == "repeat":
            o["repeat_of"] = pos[id(it["of"])] + 1
        out.append(o)
    return out


def check_items(items: list[dict], versions: list[str], rows: list[str], sets: list[str], seeds: list[int]) -> None:
    counts = {k: sum(it["kind"] == k for it in items) for k in KINDS}
    if counts != {"test": len(versions) * len(rows) * len(seeds), "same_image": len(rows), "seed_differs": len(rows),
                  "set_differs": 5, "repeat": len(rows)}:
        raise ValueError(f"unexpected counts of items: {counts}")
    tests = sorted((it["top"]["version"], it["top"]["row"], it["top"]["seed"]) for it in items if it["kind"] == "test")
    if tests != sorted((v, r, s) for v in versions for r in rows for s in seeds):
        raise ValueError("the test items do not cover every version, row and seed once")
    for n, it in enumerate(items, 1):
        t, b = it["top"], it["bottom"]
        if t["version"] != b["version"]:
            raise ValueError(f"item {n}: the two rows of an item must be of one version")
        if it["kind"] == "test" and not (t["row"] == b["row"] and t["seed"] == b["seed"] and t["author"] != b["author"]):
            raise ValueError(f"item {n}: a test item must show both authors under one seed")
        if it["kind"] == "same_image" and t != b:
            raise ValueError(f"item {n}: a same_image item must show one row twice")
        if it["kind"] == "seed_differs" and t["version"] not in GREY:
            raise ValueError(f"item {n}: a seed_differs item must be grey")
        if it["kind"] == "seed_differs" and not (t["row"] == b["row"] and t["author"] == b["author"] and t["seed"] != b["seed"]):
            raise ValueError(f"item {n}: a seed_differs item must show one author under two seeds")
        if it["kind"] == "set_differs" and not (t["row"] != b["row"] and t["row"] in sets and b["row"] in sets
                                                 and t["author"] == b["author"] and t["seed"] == b["seed"]):
            raise ValueError(f"item {n}: a set_differs item must show two sets of one author under one seed")
        if it["kind"] == "repeat":
            first = items[it["repeat_of"] - 1]
            if first["kind"] != "test" or (first["top"], first["bottom"]) != (b, t):
                raise ValueError(f"item {n}: a repeat must be a test item the other way round")
            if n - it["repeat_of"] < MIN_REPEAT_DISTANCE:
                raise ValueError(f"item {n}: the repeat is too close to its first showing")


# ---------------------------------------------------------------- spreadsheet


FILL = "FFF2CC"


def build_xlsx(images: list[bytes], q: dict, size: tuple[int, int], path: Path) -> list[str]:
    """Writes the spreadsheet; returns the fixed strings it wrote (the same for every key)."""
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as XImage
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    font, bold = Font(name="Yu Gothic", size=11), Font(name="Yu Gothic", size=11, bold=True)
    fill = PatternFill("solid", fgColor=FILL)
    wrap = Alignment(wrap_text=True, vertical="top")
    wb = Workbook()
    wb.properties.creator = ""
    wb.properties.lastModifiedBy = ""
    wb.properties.title = ""
    ws = wb.active
    ws.title = "説明"
    lines = [
        ("作品 2 のタイルの見え方の判定", True),
        ("", False),
        (f"シート「判定」の各行に、タイルの並びが上下 2 つある。{q['text']}", False),
        (f"答えは「答え」の欄の一覧から 1 つ選ぶ: {'／'.join(q['choices'])}。", False),
        (f"「一言」の欄には気づいたことを書ける（{q['comment']}）。", False),
        (q["how"], False),
        ("項目の数は {n}。記入する欄は黄色。".format(n=len(images)), False),
        ("記入例: 答え「同じに見える」、一言「明るさは違うが模様は同じ」", False),
    ]
    ws.column_dimensions["A"].width = 90
    for r, (text, b) in enumerate(lines, 1):
        c = ws.cell(row=r, column=1, value=text)
        c.font, c.alignment = (bold if b else font), wrap

    js = wb.create_sheet("判定")
    W, H = size
    js.column_dimensions["A"].width = 6
    if round(W / 7) + 2 > 255:  # the widest column Excel keeps; a wider image would cover the answers
        raise ValueError(f"an image of {W} px is too wide for one column")
    js.column_dimensions["B"].width = round(W / 7) + 2
    js.column_dimensions["C"].width = 16
    js.column_dimensions["D"].width = 40
    for col, text in zip("ABCD", ("番号", "上下の並び", "答え", "一言")):
        c = js[f"{col}1"]
        c.value, c.font = text, bold
    dv = DataValidation(type="list", formula1='"' + ",".join(q["choices"]) + '"', allow_blank=True)
    js.add_data_validation(dv)
    for i, b in enumerate(images, 1):
        r = i + 1
        js.row_dimensions[r].height = H * 0.75 + 6
        num = js.cell(row=r, column=1, value=i)
        num.font, num.alignment = font, Alignment(horizontal="center", vertical="center")
        img = XImage(io.BytesIO(b))
        img.width, img.height = W, H
        js.add_image(img, f"B{r}")
        for col in "CD":
            c = js[f"{col}{r}"]
            c.fill, c.font, c.alignment = fill, font, Alignment(wrap_text=True, vertical="center")
        dv.add(f"C{r}")
    js.freeze_panes = "A2"
    for sh in (ws, js):  # printed: landscape, one page wide
        sh.page_setup.orientation = "landscape"
        sh.page_setup.fitToWidth, sh.page_setup.fitToHeight = 1, 0
        sh.sheet_properties.pageSetUpPr.fitToPage = True
    js.print_title_rows = "1:1"
    wb.save(path)
    return [text for text, _ in lines if text] + ["番号", "上下の並び", "答え", "一言", "説明", "判定", *q["choices"]]


THEME_PART = "xl/theme/theme1.xml"
FIXED_PARTS = ("xl/sharedStrings", "xl/worksheets/sheet")  # the only parts where the fixed strings may be removed


def default_theme() -> bytes:
    from openpyxl.writer.theme import theme_xml

    return theme_xml.encode("utf-8")


def xlsx_leaks(path: Path, forbidden: list[str], forbidden_in_text: list[str], fixed: list[str] = ()) -> list[str]:
    """The forbidden strings found in the unpacked spreadsheet, after character references are resolved: `forbidden`
    anywhere in any part, `forbidden_in_text` (short tokens such as seeds) only in the strings, the properties, the
    workbook and the text and values of the cells (the drawings hold offsets in which such tokens occur by chance).
    The `fixed` strings (instructions, headers and choices, the same whatever the key) are removed only from the
    strings and the sheets before searching: a forbidden word inside them (わから in わからない) tells nothing about an
    item. The theme must be the default theme of openpyxl byte for byte (its font names hold 体 as a character
    reference) and is not searched."""
    hits = []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            raw = z.read(name)
            if name == THEME_PART:
                if raw != default_theme():
                    hits.append(f"{name}: not the default theme")
                continue
            text = html.unescape(raw.decode("utf-8", errors="strict")) if not name.startswith("xl/media/") else ""
            if name.startswith(FIXED_PARTS):
                for f in sorted(fixed, key=len, reverse=True):
                    text = text.replace(html.unescape(f), "")
            for f in forbidden:
                if f in name or f in text:
                    hits.append(f"{name}: {f}")
            if name.startswith(("xl/sharedStrings", "docProps/", "xl/workbook.xml", "xl/worksheets/sheet")):
                if name.startswith("xl/worksheets/"):
                    strings = " ".join(x.split("</t>")[0] for x in text.split("<t")[1:]) + " " + " ".join(
                        x.split("</v>")[0] for x in text.split("<v>")[1:])
                else:
                    strings = text
                for f in forbidden_in_text:  # a number only as a whole number (2006 of the namespaces holds 200)
                    pat = rf"(?<!\d){re.escape(f)}(?!\d)" if f.isdigit() else re.escape(f)
                    if re.search(pat, strings):
                        hits.append(f"{name}: {f}")
    return hits


def shown_strings(path: Path) -> set[str]:
    """Every string a cell shows, and every list of a data validation, read back with openpyxl."""
    from openpyxl import load_workbook

    wb = load_workbook(path)
    out = {cell.value for ws in wb for row in ws.iter_rows() for cell in row if isinstance(cell.value, str)}
    out |= {dv.formula1 for ws in wb for dv in ws.data_validations.dataValidation}
    return out


def anchored_images(path: Path) -> dict[int, str]:
    """Row of the sheet (1-based) -> SHA-256 of the image anchored there, read from the drawing and its relations."""
    import re

    with zipfile.ZipFile(path) as z:
        drawings = [n for n in z.namelist() if re.fullmatch(r"xl/drawings/drawing\d+\.xml", n)]
        if len(drawings) != 1:
            raise ValueError(f"expected one drawing, found {drawings}")
        d = drawings[0]
        rels = z.read(d.replace("drawings/", "drawings/_rels/") + ".rels").decode("utf-8")
        target = {}
        for rel in re.findall(r"<Relationship\b[^>]*>", rels):
            rid, tgt = re.search(r'Id="(rId\d+)"', rel).group(1), re.search(r'Target="([^"]+)"', rel).group(1)
            target[rid] = "xl/media/" + tgt.rsplit("/media/", 1)[1]
        xml = z.read(d).decode("utf-8")
        out = {}
        anchors = re.findall(r"<(?:xdr:)?oneCellAnchor>.*?</(?:xdr:)?oneCellAnchor>", xml, flags=re.S)
        if not anchors:
            raise ValueError("no image anchor found")
        if re.search(r"<(?:xdr:)?(?:twoCellAnchor|absoluteAnchor)\b", xml):
            raise ValueError("an anchor of another kind")
        linked = [n for n in z.namelist() if n.startswith("xl/worksheets/_rels/")
                  and d.rsplit("/", 1)[1] in z.read(n).decode("utf-8")]
        if linked != ["xl/worksheets/_rels/sheet2.xml.rels"]:
            raise ValueError(f"the drawing is linked from {linked}, not from the second sheet only")
        for anchor in anchors:
            frm = re.search(r"<(?:xdr:)?from>(.*?)</(?:xdr:)?from>", anchor, flags=re.S).group(1)
            pos = {k: int(re.search(rf"<(?:xdr:)?{k}>(\d+)</(?:xdr:)?{k}>", frm).group(1))
                   for k in ("col", "colOff", "row", "rowOff")}
            if (pos["col"], pos["colOff"], pos["rowOff"]) != (1, 0, 0):
                raise ValueError(f"an image is not at the top left of column B: {pos}")
            row = pos["row"] + 1
            rid = re.search(r'r:embed="(rId\d+)"', anchor).group(1)
            if row in out:
                raise ValueError(f"two images on row {row}")
            out[row] = hashlib.sha256(z.read(target[rid])).hexdigest()
    return out


def forbidden_lists(words: dict, variants: dict, authors: list[str], sets: list[str], rows: list[str],
                    versions: list[str], seeds: list[int]) -> tuple[list[str], list[str]]:
    forbidden = sorted({*words["authors"], *(a["slug"] for a in words["authors"].values()),
                        *(w["word"] for a in authors for w in words["authors"][a]["words"]),
                        *(k for k, v in variants["values"].items() if v["set"] in sets), *rows, *versions,
                        *KINDS, "seed", "swapped", "author", "own_seed"} - {"P"})
    return forbidden, sorted({str(s) for s in seeds} | {"G0", "SPC"})


def expected_shown(fixed: list[str], q: dict) -> set[str]:
    """What the cells and the list may show: the fixed strings but the sheet names, and the list of choices."""
    return (set(fixed) - {"説明", "判定"} - set(q["choices"])) | {'"' + ",".join(q["choices"]) + '"'}


def verify(path: Path) -> None:
    """The checks B4b, B5 and B5b applied again to a spreadsheet already made (nothing is written, the key is not read)."""
    c = json.loads(COND.read_bytes())
    words = json.loads((ROOT / c["inputs"]["words"]).read_bytes())
    variants = json.loads((ROOT / c["inputs"]["variants"]).read_bytes())
    forbidden, in_text = forbidden_lists(words, variants, c["authors"], c["sets"], c["rows"], list(c["versions"]),
                                         c["common_seeds"])
    from openpyxl import load_workbook

    t = c["tiles"]
    size = canvas_size(t)
    with tempfile.TemporaryDirectory() as d:  # the fixed strings as the script writes them (they do not depend on the key)
        fixed = build_xlsx([png_bytes(np.full((size[1], size[0], 3), WHITE, dtype=np.uint8))] * len(anchored_images(path)),
                           c["question"], size, Path(d) / "fixed.xlsx")
    if load_workbook(path).sheetnames != ["説明", "判定"]:
        raise SystemExit("unexpected sheets")
    leaks = xlsx_leaks(path, forbidden, in_text, fixed)
    rows = anchored_images(path)
    shown = shown_strings(path)
    if shown != expected_shown(fixed, c["question"]):
        leaks.append(f"B5b: shown strings differ: {sorted(shown ^ expected_shown(fixed, c['question']))}")
    print(json.dumps({"B5_leaks": leaks, "B5b_shown": sorted(shown), "B4b_rows": [min(rows), max(rows), len(rows),
                                                                                 len(set(rows.values()))]},
                     ensure_ascii=False, indent=1))


# ---------------------------------------------------------------- main


def main() -> None:
    for p in (OUT_DIR, REC_DIR):
        if p.exists():
            raise SystemExit(f"{p} exists; refusing to overwrite")
    raw = COND.read_bytes()
    c = json.loads(raw)
    inp = c["inputs"]
    values_path = ROOT / inp["values"]
    if cm.sha256(values_path) != inp["values_sha256"]:
        raise SystemExit("values.json differs from the one named in the conditions")
    sketch_dir = values_path.parent
    words_path, variants_path = ROOT / inp["words"], ROOT / inp["variants"]
    cm.same_as_head([COND, SKETCH_COND, WORDS_COND, values_path, sketch_dir / "sketch.json", words_path, variants_path,
                     *(ROOT / p for p in CODE)])
    state = cm.git_state(["works", "src", "results/sketch_sp_color", "results/work2_words", "results/work2_variants",
                          "pyproject.toml", "uv.lock"])
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    sk = json.loads((sketch_dir / "sketch.json").read_bytes())
    if sk["values_sha256"] != inp["values_sha256"] or sk["meta"]["conditions_sha256"] != cm.sha256(SKETCH_COND):
        raise SystemExit("values.json is not the one of the sketches on their conditions")
    wc = json.loads(WORDS_COND.read_bytes())
    cm.check_generator(wc["generator"])
    words = json.loads(words_path.read_bytes())
    variants = json.loads(variants_path.read_bytes())
    values = json.loads(values_path.read_bytes())["values"]
    eps = relief.eps_for(relief.Dimensions(**{k: wc["relief"][k] for k in ("L_mm", "D_mm", "s_max", "base_mm",
                                                                         "e_min_mm")}))
    if eps != words["eps"] or eps != sk["eps"]:
        raise SystemExit("eps differs from the records")
    D = wc["relief"]["D_mm"]
    authors, versions, rows, sets, seeds = c["authors"], list(c["versions"]), c["rows"], c["sets"], c["common_seeds"]
    if authors != wc["trial_authors"] or versions != ["G0", "P", "SPC"] or rows != [*sets, "words20"]:
        raise SystemExit("the authors, versions or rows differ from what this script draws")
    t = c["tiles"]
    kk = t["supersample"]

    def seq(row: str, a: str) -> list[str]:
        if row == "words20":
            return [w["word"] for w in words["authors"][a]["words"]]
        tl = variants["rows"][row][a]["tiles"]
        return [k for k, n in tl.items() for _ in range(n)]

    cache: dict[tuple, np.ndarray] = {}

    def tile(version: str, key: str, seed: int, px: int) -> np.ndarray:
        ck = (version, key, seed, px)
        if ck not in cache:
            v = values[key]
            pct = (2 * v["p_valence_author_equal"] - 1, 2 * v["p_arousal_author_equal"] - 1)
            V, A = (v["V"], v["A"]) if version == "G0" else pct
            sd = spelling_seed(key, seed) if version == "SPC" else seed
            g = grey_u8(z_mm(V, A, sd, px * kk, wc, eps).reshape(px, kk, px, kk).mean(axis=(1, 3)), D)
            if version == "SPC":
                cache[ck] = oklch_tint(g, hue_of(v["p_valence_author_equal"]), chroma_of(v["p_arousal_author_equal"]))[0]
            else:
                cache[ck] = np.repeat(g[..., None], 3, axis=-1)
        return cache[ck]

    # B1: the drawing is that of the sketches (every version, at their tile size and seeds, first and last tile of a line)
    skc = json.loads(SKETCH_COND.read_bytes())
    spx, sseeds = skc["drawing"]["tile_px"], skc["seeds"]
    if skc["drawing"]["supersample"] != kk or skc["rows"] != rows or skc["authors"] != authors:
        raise SystemExit("the sketches were drawn otherwise; B1 cannot compare")
    for version, fig in (("G0", "sketch_G0.png"), ("P", "sketch_P.png"), ("SPC", "sketch_SPC.png")):
        img = np.asarray(Image.open(sketch_dir / fig).convert("RGB"))
        if cm.sha256(sketch_dir / fig) != sk["figures"][fig]:
            raise SystemExit(f"{fig} differs from the record")
        for ri, row in enumerate(rows):
            for li, (s, a) in enumerate((s, a) for s in sseeds for a in authors):
                for kx in (0, len(seq(row, a)) - 1):
                    y, x = sketch_tile_place(ri, li, kx, spx)
                    if not np.array_equal(img[y : y + spx, x : x + spx], tile(version, seq(row, a)[kx], s, spx)):
                        raise SystemExit(f"B1: {version} {row} {a} {s} tile {kx} differs from the sketch")

    def row_pixels(sd: dict) -> np.ndarray:
        px = t["words20_px"] if sd["row"] == "words20" else t["sets_px"]
        return row_image([tile(sd["version"], k, sd["seed"], px) for k in seq(sd["row"], authors[sd["author"]])],
                         t["gap"])

    number = secrets.randbits(64)
    items = draw_items(random.Random(number), versions, rows, sets, seeds)
    check_items(items, versions, rows, sets, seeds)  # B6
    size = canvas_size(t)
    images = []
    for n, it in enumerate(items, 1):
        top, bot = row_pixels(it["top"]), row_pixels(it["bottom"])
        if (top.shape == bot.shape and (top == bot).all()) != (it["kind"] == "same_image"):  # B3
            raise SystemExit(f"item {n}: equal rows do not match its kind")
        png = png_bytes(compose(top, bot, size))
        ch = png_chunks(png)
        if ch[0] != "IHDR" or ch[-1] != "IEND" or set(ch[1:-1]) != {"IDAT"}:  # B4
            raise SystemExit(f"item {n}: unexpected PNG chunks {ch}")
        if Image.open(io.BytesIO(png)).size != size:
            raise SystemExit(f"item {n}: unexpected size")
        images.append(png)
    if len(set(images)) != len(images):
        raise SystemExit("two items have the same image")

    forbidden, forbidden_in_text = forbidden_lists(words, variants, authors, sets, rows, versions, seeds)

    tmp = Path(tempfile.mkdtemp(prefix=".blind-", dir=ROOT / "advice"))
    rtmp = Path(tempfile.mkdtemp(prefix=".blind_judgement-", dir=ROOT / "results"))
    try:
        xlsx = tmp / "judgement.xlsx"
        fixed = build_xlsx(images, c["question"], size, xlsx)
        leaks = xlsx_leaks(xlsx, forbidden, forbidden_in_text, fixed)  # B5
        if leaks:
            raise SystemExit(f"the spreadsheet reveals: {leaks}")
        if shown_strings(xlsx) != expected_shown(fixed, c["question"]):  # B5b
            raise SystemExit("the spreadsheet shows strings other than the fixed ones")
        if anchored_images(xlsx) != {n + 1: hashlib.sha256(b).hexdigest() for n, b in enumerate(images, 1)}:  # B4b
            raise SystemExit("the image on the row of an item is not the image of that item")
        (tmp / "sealed").mkdir()
        key = {"conditions_sha256": hashlib.sha256(raw).hexdigest(), "random_number": number,
               "authors": authors, "items": [{"number": n, **it, "image_sha256": hashlib.sha256(b).hexdigest()}
                                            for n, (it, b) in enumerate(zip(items, images), 1)],
               "spreadsheet_sha256": cm.sha256(xlsx)}
        with (tmp / "sealed" / "key.json").open("x") as fh:
            fh.write(json.dumps(key, indent=1, ensure_ascii=False) + "\n")
        record = {"meta": {"conditions_sha256": key["conditions_sha256"], "script_sha256": cm.sha256(Path(__file__)),
                           **cm.versions("numpy", "scipy", "pillow", "openpyxl"), **state,
                           "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
                  "spreadsheet_sha256": key["spreadsheet_sha256"], "key_sha256": cm.sha256(tmp / "sealed" / "key.json"),
                  "n_items": len(items), "canvas": list(size),
                  "checks": {"B1": True, "B3": True, "B4": True, "B4b": True, "B5": True, "B5b": True, "B6": True}}
        with (rtmp / "record.json").open("x") as fh:
            fh.write(json.dumps(record, indent=1) + "\n")
    except BaseException:
        shutil.rmtree(tmp)
        shutil.rmtree(rtmp)
        raise
    if OUT_DIR.exists() or REC_DIR.exists():
        raise SystemExit(f"an output appeared during the run; left in {tmp} and {rtmp}")
    tmp.rename(OUT_DIR)
    rtmp.rename(REC_DIR)
    print(json.dumps({"spreadsheet_sha256": record["spreadsheet_sha256"], "key_sha256": record["key_sha256"]}, indent=1))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--verify"] and len(sys.argv) == 3:
        verify(Path(sys.argv[2]))
    elif sys.argv[1:]:
        raise SystemExit(f"unknown arguments {sys.argv[1:]}")
    else:
        main()
