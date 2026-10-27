"""Tests of the recount by spelling and of the tiles per set of works/work2_variants.py on lemmas and counts written for
the tests (no text of any work, no lexicon)."""

from __future__ import annotations

import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
sys.path.insert(0, str(ROOT / "checks"))

from context_choice import match_set  # noqa: E402
from work2_variants import allocate, by_spelling, entry_of, figure, row_of, spellings, to_variants  # noqa: E402

ADV = {"pos1": ["接続詞"], "variants": {"しかし": [["しかし"], ["然し"]], "だが": [["だが"]]}}
INT = {"pos1": ["副詞", "名詞"], "variants": {"とても": [["とても"]], "非常に": [["非常", "に"]]}}


def test_spellings_keep_the_order_and_join_two_lemmas():
    assert spellings(INT) == [("とても", "とても", ("とても",)), ("非常に", "非常に", ("非常", "に"))]
    assert [k for _, k, _ in spellings(ADV)] == ["しかし", "然し", "だが"]


def test_recount_by_spelling_sums_to_the_variants():
    toks = [("然し", "接続詞", "*"), ("しかし", "接続詞", "*"), ("然し", "接続詞", "*"), ("だが", "接続詞", "*"),
            ("だが", "副詞", "一般")]
    per = match_set(toks, by_spelling(ADV))
    assert per == {"然し": 2, "しかし": 1, "だが": 1}
    assert to_variants(per, ADV) == match_set(toks, ADV)


def test_recount_keeps_two_lemma_spellings_and_parts_of_speech():
    toks = [("非常", "名詞", "形容動詞語幹"), ("に", "助詞", "副詞化"), ("非常", "名詞", "形容動詞語幹"),
            ("な", "助動詞", "*"), ("とても", "副詞", "一般")]
    per = match_set(toks, by_spelling(INT))
    assert per == {"非常に": 1, "とても": 1}
    assert to_variants(per, INT) == match_set(toks, INT)


def test_pos2_is_carried_into_the_spec_by_spelling():
    first = {"pos1": ["名詞"], "pos2": ["代名詞"], "variants": {"私": [["私"], ["わたし"]]}}
    assert by_spelling(first)["pos2"] == ["代名詞"]
    assert match_set([("私", "名詞", "代名詞"), ("私", "名詞", "一般")], by_spelling(first)) == {"私": 1}


def test_entry_of_falls_back_to_the_first_lemma_only_for_two_lemmas():
    entries = {"非常": (0, 0), "しかし": (0, 0)}
    assert entry_of("しかし", ("しかし",), entries) == "しかし"
    assert entry_of("非常に", ("非常", "に"), entries) == "非常"
    assert entry_of("然し", ("然し",), entries) is None


def test_largest_remainder_sums_to_n_and_breaks_ties_by_order():
    order = ["a", "b", "c"]
    assert allocate({"a": 1, "b": 1, "c": 1}, order, 10) == {"a": 4, "b": 3, "c": 3}
    assert allocate({"a": 56, "b": 23, "c": 21}, order, 10) == {"a": 6, "b": 2, "c": 2}
    assert allocate({"a": 96, "b": 4}, ["a", "b"], 10) == {"a": 10, "b": 0}  # a share of 0.04 disappears
    for counts in ({"a": 7, "b": 2, "c": 1}, {"a": 1, "c": 13}, {"b": 5}):
        assert sum(allocate(counts, order, 10).values()) == 10


def test_equal_remainders_are_ties_broken_by_order():
    # the three remainders are 2/3 exactly; in floating point they differ by an ulp and the order was lost
    assert allocate({"a": 2, "b": 2, "c": 8}, ["a", "b", "c"], 10) == {"a": 2, "b": 2, "c": 6}


def test_fractions_are_exact_and_floats_are_refused():
    a, b = (Fraction(7, 10) + Fraction(1, 10)) / 2, (Fraction(3, 10) + Fraction(9, 10)) / 2
    assert allocate({"a": a, "b": b}, ["a", "b"], 10) == {"a": 4, "b": 6}
    with pytest.raises(TypeError):
        allocate({"a": 0.4, "b": 0.6}, ["a", "b"], 10)


def test_two_spellings_with_the_same_key_stop():
    clash = {"pos1": ["名詞"], "variants": {"x": [["非常", "に"]], "y": [["非常に"]]}}
    with pytest.raises(ValueError):
        by_spelling(clash)


def test_figure_puts_the_tiles_in_order_and_uses_the_seed_of_the_block():
    f = {"tile_px": 4, "gap_px": 1, "tile_label_px": 2, "row_label_px": 2, "row_gap_px": 1, "block_gap_px": 3,
         "font_file": "/System/Library/Fonts/Hiragino Sans GB.ttc", "n_tiles": 3}
    level = {("a", 1): -0.5, ("b", 1): 0.0, ("a", 2): 0.5, ("b", 2): 0.25}

    def tile_of(key, seed):
        return np.full((4, 4), level[key, seed])

    img = np.asarray(figure([("", [("", ["a", "b", "b"]), ("", None)], 1), ("", [("", ["a", "b", "b"])], 2)],
                            tile_of, f, 1.0))
    y = 36 + 2 + 1  # head, row label, inside the tile
    bw = 3 * 4 + 2 * 1
    got = [[int(img[y, x0 + k * 5 + 1]) for k in range(3)] for x0 in (0, bw + 3)]
    assert got == [[0, 128, 128], [255, 191, 191]]


def test_row_with_too_few_is_not_drawn_and_keeps_its_n():
    row = row_of(Counter({"a": 4, "b": 1}), {"a": Fraction(4, 5), "b": Fraction(1, 5)}, ["a", "b"], 10, (1, 4))
    assert row["too_few"] and row["tiles"] is None and row["n"] == 5 and row["works"] == [1, 4]


def test_row_reports_zero_tiles_and_equal_weight():
    row = row_of(Counter({"a": 96, "b": 4}), {"a": Fraction(1, 2), "b": Fraction(1, 2)}, ["a", "b", "c"], 10, (2, 4))
    assert row["tiles"] == {"a": 10} and row["zero_tiles"] == ["b"]
    assert row["equal_weight_tiles"] == {"a": 5, "b": 5} and row["equal_weight_differs"]
    assert row["share"] == {"a": 0.96, "b": 0.04}


def test_empty_row_has_no_tiles():
    row = row_of(Counter(), {}, ["a"], 10, (0, 3))
    assert row["too_few"] and row["n"] == 0


@pytest.mark.parametrize("n", [10, 20])
def test_allocation_is_monotone_in_the_share(n):
    order = ["a", "b"]
    got = [allocate({"a": k, "b": 100 - k}, order, n)["a"] for k in range(0, 101)]
    assert got == sorted(got)
