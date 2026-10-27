"""Tests of the differences, the rule of figure 6 and the sheets of works/work2_seed_check.py on arrays written for the
tests (no word, no lexicon)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
sys.path.insert(0, str(ROOT / "checks"))

from work2_seed_check import (  # noqa: E402
    answers_to_rule, check_answers, figure6, grey_u8, ratios, row_diff, sheet, tile_diff, tiles_differing)

SETS = ["L1", "L2", "P1", "O1"]


def test_tile_and_row_differences_are_scaled_by_D():
    a, b = np.zeros((4, 4)), np.full((4, 4), 0.6)
    assert tile_diff(a, b, 1.2) == pytest.approx(0.5)
    assert row_diff([a, a], [a, b], 1.2) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        row_diff([a], [a, b], 1.2)
    with pytest.raises(ValueError):
        tile_diff(a, np.zeros((3, 3)), 1.2)


def test_tiles_differing_counts_the_tiles_not_shared():
    assert tiles_differing({"a": 6, "b": 4}, {"a": 6, "b": 4}) == 0
    assert tiles_differing({"a": 10}, {"b": 10}) == 10
    assert tiles_differing({"a": 2, "b": 6, "c": 2}, {"a": 4, "d": 4, "b": 1, "c": 1}) == 6
    with pytest.raises(ValueError):
        tiles_differing({"a": 9}, {"a": 10})


def test_ratio_uses_only_the_seed_pairs_that_contain_the_seed():
    bs = {("x", 1, 2): 1.0, ("y", 1, 2): 2.0, ("x", 1, 3): 3.0, ("y", 1, 3): 9.0, ("x", 2, 3): 100.0, ("y", 2, 3): 200.0}
    r = ratios({1: 1.0, 2: 1.0, 3: 1.0}, bs)
    assert r[1] == pytest.approx(1 / 2.5)  # median of 1, 2, 3, 9 (both authors)
    assert r[2] == pytest.approx(1 / 51.0)  # median of 1, 2, 100, 200
    with pytest.raises(SystemExit):
        ratios({1: 1.0}, {("x", 1, 2): 0.0})


def test_figure6_takes_the_largest_remaining_and_the_smallest_invisible():
    med = {"L1": 0.02, "L2": 0.05, "P1": 0.09, "O1": 0.01}
    diff = {"L1": 4, "L2": 3, "P1": 5, "O1": 7}
    got = figure6(SETS, {"L1": False, "L2": True, "P1": True, "O1": False},
                  {"L1": True, "L2": False, "P1": False, "O1": True}, med, diff)
    assert got == {"visible": "P1", "visible_remains": True, "invisible": "O1", "invisible_judged": True,
                   "invisible_eligible": SETS}


def test_figure6_falls_back_when_nothing_remains_and_respects_the_three_tiles():
    med = {"L1": 0.02, "L2": 0.05, "P1": 0.09, "O1": 0.001}
    diff = {"L1": 4, "L2": 3, "P1": 5, "O1": 2}  # O1 differs by 2 tiles only: not eligible as the invisible example
    none = {s: False for s in SETS}
    got = figure6(SETS, none, none, med, diff)
    assert got == {"visible": "P1", "visible_remains": False, "invisible": "L1", "invisible_judged": False,
                   "invisible_eligible": ["L1", "L2", "P1"]}
    few = figure6(SETS, none, none, med, {s: 2 for s in SETS})
    assert few["invisible"] is None and few["invisible_eligible"] == []


def test_figure6_ties_go_by_the_order_of_the_sets():
    med = {s: 0.05 for s in SETS}
    diff = {s: 5 for s in SETS}
    yes = {s: True for s in SETS}
    no = {s: False for s in SETS}
    assert figure6(SETS, yes, no, med, diff)["visible"] == "L1"
    assert figure6(SETS, no, yes, med, diff)["invisible"] == "L1"


def test_answers_must_cover_every_common_seed_with_booleans():
    a = {"L1": {"100": True, "200": True}, "L2": {"100": True, "200": False}}
    rem, inv = answers_to_rule(a, ["L1", "L2"], [100, 200])
    assert rem == {"L1": True, "L2": False} and inv == {"L1": False, "L2": False}
    with pytest.raises(SystemExit):
        answers_to_rule({"L1": {"100": True}}, ["L1"], [100, 200])
    with pytest.raises(SystemExit):
        answers_to_rule({"L1": {"100": 1, "200": True}}, ["L1"], [100, 200])


def test_sheet_has_no_text_and_puts_the_second_author_after_the_gap():
    D = 1.0
    lo, hi = np.full((8, 8), -0.5), np.full((8, 8), 0.5)
    img = np.asarray(sheet([([lo, lo], [hi, hi]), ([hi], [lo])], 8, D, gap=2, pair_gap=10, line_gap=4))
    assert img.shape == (8 * 2 + 4, 4 * 8 + 2 * 2 + 10)
    assert img[2, 2] == 0 and img[2, 2 * 8 + 2 + 10 + 2] == 255  # first line: author 1 dark, author 2 light
    assert img[14, 2] == 255 and img[14, 2 * 8 + 2 + 10 + 2] == 0  # second line, one tile each, same columns
    assert set(np.unique(img[8:12])) == {255}  # the gap between the lines is blank


def test_answers_file_needs_three_keys_and_deciding_equal_to_the_researcher():
    rows, seeds = ["L1", "words20"], [100, 200]
    yes = {r: {"100": True, "200": True} for r in rows}
    mixed = {"L1": {"100": True, "200": False}, "words20": {"100": True, "200": True}}
    check_answers({"researcher": yes, "main": mixed, "deciding": yes}, rows, seeds)
    with pytest.raises(SystemExit):  # deciding took the main's answer
        check_answers({"researcher": yes, "main": mixed, "deciding": mixed}, rows, seeds)
    with pytest.raises(SystemExit):  # a key is missing
        check_answers({"researcher": yes, "deciding": yes}, rows, seeds)
    with pytest.raises(SystemExit):  # words20 is not answered
        check_answers({k: {"L1": yes["L1"]} for k in ("researcher", "main", "deciding")}, rows, seeds)


def test_grey_puts_the_origin_at_the_lower_left():
    z = np.zeros((4, 4))
    z[0, 0] = 0.5  # row 0 is y = 0, the bottom of the tile
    img = grey_u8(z, 1.0)
    assert img[3, 0] == 255 and img[0, 0] == 128
