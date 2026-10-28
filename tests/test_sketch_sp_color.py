"""Tests of the percentiles and the colour of works/sketch_sp_color.py on values written for the tests."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))

from sketch_sp_color import (  # noqa: E402
    chroma_of, hue_of, linear_to_oklab, oklab_to_linear, oklch_tint, reference_weights, srgb_to_linear,
    weighted_percentile)


def test_weighted_percentile_counts_ties_as_half_of_their_weight():
    v, w = np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 1.0])
    assert weighted_percentile(v, w, 0.0) == 0.0
    assert weighted_percentile(v, w, 2.0) == pytest.approx((1 + 1) / 4)
    assert weighted_percentile(v, w, 9.0) == 1.0
    with pytest.raises(ValueError):
        weighted_percentile(v, np.zeros(3), 2.0)


def test_reference_weights_give_each_author_weight_one():
    w = reference_weights({"a": Counter({"x": 3, "y": 1}), "b": Counter({"x": 1, "z": 9})})
    assert sum(d["author_equal"] for d in w.values()) == pytest.approx(2.0)
    assert w["x"]["author_equal"] == pytest.approx(3 / 4 + 1 / 10)
    assert w["x"]["tokens"] == 4 and w["z"]["types"] == 1.0
    with pytest.raises(ValueError):
        reference_weights({"a": Counter()})


def test_oklab_round_trip():
    rgb = np.random.default_rng(0).uniform(0, 1, (50, 3))
    assert np.allclose(oklab_to_linear(linear_to_oklab(rgb)), rgb, atol=1e-9)


def test_oklch_tint_keeps_lightness_and_stays_in_gamut():
    grey = np.array([[0, 40, 128, 200, 255]], dtype=np.uint8)
    rgb, share = oklch_tint(grey, hue_of(1.0), chroma_of(1.0))
    lab = linear_to_oklab(srgb_to_linear(rgb.astype(float) / 255))
    lg = linear_to_oklab(np.repeat(srgb_to_linear(grey.astype(float) / 255)[..., None], 3, axis=-1))[..., 0]
    assert np.allclose(lab[..., 0], lg, atol=0.01)
    assert 0 < share <= 1  # black and white cannot keep any chroma
    mid, s_mid = oklch_tint(np.array([[128]], dtype=np.uint8), hue_of(0.0), chroma_of(0.0))
    assert s_mid == 0.0 and mid[0, 0, 2] > mid[0, 0, 0]  # blue at the unpleasant end


def test_hue_path_avoids_green():
    hues = [hue_of(p) % 360 for p in np.linspace(0, 1, 101)]
    assert all(not (100 < h < 200) for h in hues)
