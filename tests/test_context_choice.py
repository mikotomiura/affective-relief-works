"""Tests of the matching of check 6 on lemmas written for the tests (no text of any work)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "checks"))
sys.path.insert(0, str(ROOT / "works"))

from context_choice import analysable, match_set  # noqa: E402

SUDDEN = {"pos1": ["副詞", "名詞"], "variants": {"突然": [["突然"]], "急に": [["急", "に"]]}}
ADV = {"pos1": ["接続詞"], "variants": {"しかし": [["しかし"], ["然し"]], "だが": [["だが"]]}}
FIRST = {"pos1": ["名詞"], "pos2": ["代名詞"], "variants": {"私": [["私"]]}}


def test_two_lemma_variant_and_part_of_speech():
    toks = [("急", "名詞", "形容動詞語幹"), ("に", "助詞", "副詞化"), ("突然", "副詞", "助詞類接続"),
            ("急", "名詞", "形容動詞語幹"), ("な", "助動詞", "*")]
    assert match_set(toks, SUDDEN) == {"急に": 1, "突然": 1}  # 急な is not 急に


def test_notation_variants_merge_and_pos_is_checked():
    toks = [("然し", "接続詞", "*"), ("しかし", "接続詞", "*"), ("だが", "接続詞", "*"), ("だが", "副詞", "一般")]
    assert match_set(toks, ADV) == {"しかし": 2, "だが": 1}


def test_pos2_restricts():
    toks = [("私", "名詞", "代名詞"), ("私", "名詞", "一般")]
    assert match_set(toks, FIRST) == {"私": 1}


def test_analysable_needs_eight_works_and_three_authors_with_two():
    assert analysable(np.array([0, 0, 1, 1, 2, 2, 3, 4]))
    assert not analysable(np.array([0, 0, 1, 1, 2, 3, 4]))  # seven works
    assert not analysable(np.array([0, 0, 1, 1, 2, 3, 4, 5]))  # two authors with two works


def test_scene_carries_open_quotations():
    from context_choice import scenes_of

    sents = ["「どうした。", "行こう」と言った。", "彼は「はい」と答えた。", "雨が降る。", "――「いや」"]
    assert scenes_of(sents) == ["dialogue", "dialogue", "mixed", "narration", "mixed"]


def test_scene_of_body_counts_the_marks_of_dropped_pieces():
    from context_choice import scenes_of, scenes_of_body

    body = "「待て。\n……」\n雨が降る。"
    # the kept sentences are 「待て。 and 雨が降る。; the dropped ……」 closes the quotation
    assert scenes_of(["「待て。", "雨が降る。"]) == ["dialogue", "dialogue"]  # version 1: stuck
    assert scenes_of_body(body) == ["dialogue", "narration"]
