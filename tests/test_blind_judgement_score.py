"""Tests of the reading and the scoring of works/blind_judgement_score.py on items and answers written for the tests."""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))

from blind_judgement import draw_items  # noqa: E402
from blind_judgement_score import DIFFERENT, SAME, UNSURE, agreement, decide, normalise, score_judge  # noqa: E402

VERSIONS = ["G0", "P", "SPC"]
ROWS = ["L1", "L2", "P1", "O1", "words20"]
SEEDS = [100, 200, 300, 400, 500]


def items() -> list[dict]:
    return [{"number": n, **it} for n, it in enumerate(draw_items(random.Random(7), VERSIONS, ROWS, ROWS[:4], SEEDS), 1)]


def answer_all(its: list[dict], test_rule) -> dict[int, str | None]:
    out = {}
    for it in its:
        k = it["kind"]
        out[it["number"]] = (SAME if k == "same_image" else DIFFERENT if k == "seed_differs"
                             else test_rule(it) if k in ("test", "repeat") else UNSURE)
    return out


def test_normalise_accepts_only_the_choices():
    assert normalise(None) is None and normalise("  ") is None
    assert normalise(" 同じに見える ") == SAME
    with pytest.raises(ValueError):
        normalise("違う")


def test_a_row_remains_only_if_different_under_every_seed():
    its = items()
    ans = answer_all(its, lambda it: DIFFERENT if it["top"]["version"] == "SPC" or it["top"]["row"] == "P1" else SAME)
    s = score_judge(its, ans, VERSIONS, ROWS, SEEDS)
    assert s["controls"]["pass"]
    d = decide(s, ROWS)
    assert d["rows_remaining"] == {"G0": ["P1"], "P": ["P1"], "SPC": ROWS} and d["two_claims"]
    one = next(it for it in its if it["kind"] == "test" and it["top"]["version"] == "G0" and it["top"]["row"] == "P1")
    ans[one["number"]] = UNSURE  # わからない under one seed: the row no longer remains
    d = decide(score_judge(its, ans, VERSIONS, ROWS, SEEDS), ROWS)
    assert d["rows_remaining"]["G0"] == [] and not d["two_claims"]


def test_controls_decide_whether_the_rule_is_applied():
    its = items()
    ans = answer_all(its, lambda it: DIFFERENT)
    first_same = next(it["number"] for it in its if it["kind"] == "same_image")
    ans[first_same] = None  # an empty control answer fails the controls
    s = score_judge(its, ans, VERSIONS, ROWS, SEEDS)
    assert not s["controls"]["pass"] and decide(s, ROWS)["decided"] is False
    ans = answer_all(its, lambda it: DIFFERENT)
    for n in [it["number"] for it in its if it["kind"] == "seed_differs"][:2]:
        ans[n] = SAME  # 3 of 5 is not enough
    assert not score_judge(its, ans, VERSIONS, ROWS, SEEDS)["controls"]["pass"]


def test_repeats_and_agreement_are_reported():
    its = items()
    ans = answer_all(its, lambda it: DIFFERENT)
    rep = next(it for it in its if it["kind"] == "repeat")
    ans[rep["number"]] = SAME
    s = score_judge(its, ans, VERSIONS, ROWS, SEEDS)
    assert sum(not r["agree"] for r in s["repeats"]) == 1
    other = dict(ans)
    for it in its:
        if it["kind"] == "test":
            other[it["number"]] = SAME
            break
    assert agreement(its, ans, other) == {"test_items": 75, "answered_alike": 74}


def test_reply_checks_find_changed_cells_and_lost_images(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    from blind_judgement import build_xlsx, canvas_size, png_bytes
    from blind_judgement_score import check_reply, read_answers

    t = {"sets_px": 12, "words20_px": 8, "supersample": 1, "gap": 4}
    size = canvas_size(t)
    imgs = [png_bytes(np.full((size[1], size[0], 3), k, dtype=np.uint8)) for k in range(3)]
    sent = tmp_path / "sent.xlsx"
    build_xlsx(imgs, {"text": "a", "choices": [DIFFERENT, SAME, UNSURE], "comment": "c", "how": "h"}, size, sent)
    same = tmp_path / "same.xlsx"
    shutil.copy(sent, same)
    assert check_reply(sent, same, 3) == []
    assert read_answers(same, 3) == ({1: None, 2: None, 3: None}, {})
    wb = openpyxl.load_workbook(sent)
    wb["判定"]["C2"] = SAME
    wb["判定"]["D3"] = "note"
    wb.save(tmp_path / "answered.xlsx")  # only the cells to fill in changed (openpyxl keeps the images)
    assert check_reply(sent, tmp_path / "answered.xlsx", 3) == []
    assert read_answers(tmp_path / "answered.xlsx", 3) == ({1: SAME, 2: None, 3: None}, {2: "note"})
    wb = openpyxl.load_workbook(tmp_path / "answered.xlsx")  # a workbook with images saves only once
    wb["説明"]["A1"] = "changed"
    wb.save(tmp_path / "edited.xlsx")
    assert any("説明" in p for p in check_reply(sent, tmp_path / "edited.xlsx", 3))
    import zipfile
    with zipfile.ZipFile(sent) as zi, zipfile.ZipFile(tmp_path / "image.xlsx", "w") as zo:
        for n in zi.namelist():
            zo.writestr(n, png_bytes(np.full((size[1], size[0], 3), 9, dtype=np.uint8)) if n == "xl/media/image2.png"
                        else zi.read(n))
    assert any("images" in p for p in check_reply(sent, tmp_path / "image.xlsx", 3))


def test_reply_checks_read_relationship_ids_of_any_form(tmp_path):
    """Other applications than Excel write relationship ids such as R05dbda31c4d241ab instead of rId1."""
    pytest.importorskip("openpyxl")
    import re
    import zipfile
    from blind_judgement import build_xlsx, canvas_size, png_bytes
    from blind_judgement_score import anchored_pixels, check_reply

    t = {"sets_px": 12, "words20_px": 8, "supersample": 1, "gap": 4}
    size = canvas_size(t)
    imgs = [png_bytes(np.full((size[1], size[0], 3), k, dtype=np.uint8)) for k in range(3)]
    sent = tmp_path / "sent.xlsx"
    build_xlsx(imgs, {"text": "a", "choices": [DIFFERENT, SAME, UNSURE], "comment": "c", "how": "h"}, size, sent)
    other = tmp_path / "other.xlsx"
    rename = lambda b: re.sub(rb'"rId(\d+)"', lambda m: b'"R0a1b2c3d' + m.group(1) + b'"', b)  # noqa: E731
    with zipfile.ZipFile(sent) as zi, zipfile.ZipFile(other, "w") as zo:
        for n in zi.namelist():
            zo.writestr(n, rename(zi.read(n)) if n.startswith("xl/drawings/") else zi.read(n))
    with zipfile.ZipFile(other) as z:
        assert b'"R0a1b2c3d1"' in z.read("xl/drawings/_rels/drawing1.xml.rels")
    assert len(anchored_pixels(other)) == 3
    assert anchored_pixels(other) == anchored_pixels(sent)
    assert check_reply(sent, other, 3) == []
