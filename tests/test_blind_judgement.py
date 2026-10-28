"""Tests of the rows, the draws, the images and the spreadsheet of works/blind_judgement.py on arrays written for the
tests (no word, no lexicon)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))

from blind_judgement import (  # noqa: E402
    MIN_REPEAT_DISTANCE, canvas_size, check_items, compose, draw_items, png_bytes, png_chunks, row_image,
    sketch_tile_place)

T = {"sets_px": 12, "words20_px": 8, "supersample": 1, "gap": 4}
VERSIONS = ["G0", "P", "SPC"]
ROWS = ["L1", "L2", "P1", "O1", "words20"]
SETS = ROWS[:4]
SEEDS = [100, 200, 300, 400, 500]


def test_row_image_places_tiles_with_white_gaps():
    tiles = [np.full((3, 3, 3), k, dtype=np.uint8) for k in range(3)]
    r = row_image(tiles, 2)
    assert r.shape == (3, 3 * 3 + 2 * 2, 3)
    assert (r[:, 3:5] == 255).all() and (r[:, 5:8] == 1).all() and (r[:, 10:13] == 2).all()


def test_sketch_tile_place_follows_the_figure_layout():
    assert sketch_tile_place(0, 0, 0, 72) == (22, 150)
    assert sketch_tile_place(0, 1, 2, 72) == (22 + 75, 150 + 2 * 75)
    assert sketch_tile_place(1, 0, 0, 72) == (22 + 4 * 75 + 18 + 22, 150)


@pytest.mark.parametrize("number", range(20))
def test_draws_pass_the_checks(number):
    items = draw_items(random.Random(number), VERSIONS, ROWS, SETS, SEEDS)
    assert len(items) == 95
    check_items(items, VERSIONS, ROWS, SETS, SEEDS)
    for n, it in enumerate(items, 1):
        if it["kind"] == "repeat":
            assert n - it["repeat_of"] >= MIN_REPEAT_DISTANCE


def test_draws_put_both_authors_on_top_sometimes():
    tops = [it["top"]["author"] for n in range(5) for it in draw_items(random.Random(n), VERSIONS, ROWS, SETS, SEEDS)
            if it["kind"] == "test"]
    assert 0 < sum(tops) < len(tops)


def test_check_items_rejects_broken_items():
    items = draw_items(random.Random(3), VERSIONS, ROWS, SETS, SEEDS)

    def broken(kind: str, change) -> list[dict]:
        i = next(i for i, it in enumerate(items) if it["kind"] == kind)
        out = [dict(it) for it in items]
        out[i] = change(dict(out[i]))
        return out

    for bad in (
        broken("test", lambda it: {**it, "bottom": dict(it["top"])}),
        broken("repeat", lambda it: {**it, "top": it["bottom"], "bottom": it["top"]}),
        broken("same_image", lambda it: {**it, "bottom": {**it["top"], "author": 1 - it["top"]["author"]}}),
        broken("test", lambda it: {**it, "bottom": {**it["bottom"], "version": "P" if it["top"]["version"] != "P" else "G0"}}),
        items[:-1],
    ):
        with pytest.raises(ValueError):
            check_items(bad, VERSIONS, ROWS, SETS, SEEDS)


def test_compose_places_rows_on_one_canvas_size():
    size = canvas_size(T)
    assert size == (20 * 8 + 19 * 4, 2 * 12 + 24)
    w = np.zeros((8, 20 * 8 + 19 * 4, 3), dtype=np.uint8)
    s = np.full((12, 10 * 12 + 9 * 4, 3), 7, dtype=np.uint8)
    a = compose(s, w, size)
    assert a.shape == (size[1], size[0], 3)
    assert (a[:12, : s.shape[1]] == 7).all() and (a[:12, s.shape[1] :] == 255).all()
    assert (a[12 + 24 + 2 : 12 + 24 + 10, : w.shape[1]] == 0).all()
    assert (a[12 : 12 + 24] == 255).all()
    with pytest.raises(ValueError):
        compose(np.zeros((20, 5, 3), dtype=np.uint8), s, size)


def test_png_has_no_text_chunks():
    chunks = png_chunks(png_bytes(np.zeros((4, 4, 3), dtype=np.uint8)))
    assert chunks[0] == "IHDR" and chunks[-1] == "IEND" and set(chunks[1:-1]) == {"IDAT"}


def test_spreadsheet_has_the_images_and_the_leak_check_finds_planted_strings(tmp_path):
    pytest.importorskip("openpyxl")
    from blind_judgement import build_xlsx, xlsx_leaks

    size = canvas_size(T)
    imgs = [png_bytes(np.full((size[1], size[0], 3), k, dtype=np.uint8)) for k in range(3)]
    q = {"text": "上下は違って見えるか。", "choices": ["違って見える", "同じに見える", "わからない"], "comment": "任意で一言",
         "how": "拡大してかまわない。"}
    p = tmp_path / "a.xlsx"
    build_xlsx(imgs, q, size, p)
    assert xlsx_leaks(p, ["作家甲", "seed", "words20"], ["100", "SPC"]) == []
    planted = tmp_path / "b.xlsx"
    build_xlsx(imgs, {**q, "how": "作家甲 の seed は 100 で SPC"}, size, planted)
    hits = xlsx_leaks(planted, ["作家甲", "seed"], ["100", "SPC"])
    assert all(any(f in h for h in hits) for f in ("作家甲", "seed", "100", "SPC"))


def test_leak_check_masks_the_fixed_strings_and_maps_rows_to_images(tmp_path):
    pytest.importorskip("openpyxl")
    import json

    from blind_judgement import anchored_images, build_xlsx, xlsx_leaks

    q = json.loads((ROOT / "works" / "conditions" / "blind_judgement.json").read_bytes())["question"]
    size = canvas_size(T)
    imgs = [png_bytes(np.full((size[1], size[0], 3), k, dtype=np.uint8)) for k in range(4)]
    p = tmp_path / "a.xlsx"
    fixed = build_xlsx(imgs, q, size, p)
    assert xlsx_leaks(p, ["わから", "全部"], []) != []  # the choice holds a forbidden word
    assert xlsx_leaks(p, ["わから", "全部"], [], fixed) == []
    import hashlib
    assert anchored_images(p) == {n + 1: hashlib.sha256(b).hexdigest() for n, b in enumerate(imgs, 1)}


def test_numbers_are_matched_as_whole_numbers(tmp_path):
    pytest.importorskip("openpyxl")
    from blind_judgement import build_xlsx, xlsx_leaks

    size = canvas_size(T)
    imgs = [png_bytes(np.full((size[1], size[0], 3), 1, dtype=np.uint8))]
    q = {"text": "a", "choices": ["x"], "comment": "c", "how": "h 2006 1000"}
    p = tmp_path / "a.xlsx"
    build_xlsx(imgs, q, size, p)
    assert xlsx_leaks(p, [], ["200", "100"]) == []
    build_xlsx(imgs, {**q, "how": "h 200"}, size, p)
    assert xlsx_leaks(p, [], ["200"]) != []


def _rewrite(src: Path, dst: Path, part: str, change) -> None:
    import zipfile

    with zipfile.ZipFile(src) as zi, zipfile.ZipFile(dst, "w") as zo:
        for n in zi.namelist():
            b = zi.read(n)
            zo.writestr(n, change(b) if n == part else b)


def test_leak_check_resolves_character_references_and_guards_the_theme(tmp_path):
    pytest.importorskip("openpyxl")
    from blind_judgement import THEME_PART, build_xlsx, expected_shown, shown_strings, xlsx_leaks

    size = canvas_size(T)
    imgs = [png_bytes(np.full((size[1], size[0], 3), 1, dtype=np.uint8))]
    q = {"text": "a", "choices": ["x", "y"], "comment": "c", "how": "h"}
    p = tmp_path / "a.xlsx"
    fixed = build_xlsx(imgs, q, size, p)
    assert shown_strings(p) == expected_shown(fixed, q)
    assert xlsx_leaks(p, ["体"], [], fixed) == []  # 体 is in the default theme only, as a character reference
    hidden = tmp_path / "b.xlsx"  # 作家 written as character references in the properties
    _rewrite(p, hidden, "docProps/core.xml", lambda b: b.replace(b"</cp:coreProperties>",
                                                                  b"<dc:subject>&#x4F5C;&#x5BB6;</dc:subject></cp:coreProperties>"))
    assert any("作家" in h for h in xlsx_leaks(hidden, ["作家"], [], fixed))
    theme = tmp_path / "c.xlsx"
    _rewrite(p, theme, THEME_PART, lambda b: b.replace(b"Calibri", b"Calibra", 1))
    assert any("theme" in h for h in xlsx_leaks(theme, [], [], fixed))


def test_image_check_rejects_an_image_out_of_column_b(tmp_path):
    pytest.importorskip("openpyxl")
    from blind_judgement import anchored_images, build_xlsx

    size = canvas_size(T)
    imgs = [png_bytes(np.full((size[1], size[0], 3), k, dtype=np.uint8)) for k in range(2)]
    p = tmp_path / "a.xlsx"
    build_xlsx(imgs, {"text": "a", "choices": ["x"], "comment": "c", "how": "h"}, size, p)
    moved = tmp_path / "b.xlsx"
    _rewrite(p, moved, "xl/drawings/drawing1.xml", lambda b: b.replace(b"<col>1</col>", b"<col>2</col>", 1))
    with pytest.raises(ValueError):
        anchored_images(moved)
    shifted = tmp_path / "c.xlsx"
    _rewrite(p, shifted, "xl/drawings/drawing1.xml", lambda b: b.replace(b"<rowOff>0</rowOff>", b"<rowOff>9</rowOff>", 1))
    with pytest.raises(ValueError):
        anchored_images(shifted)
