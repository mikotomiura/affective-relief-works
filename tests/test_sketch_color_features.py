"""Tests of the features and the tint of works/sketch_color_features.py on values written for the tests."""

from __future__ import annotations

import colorsys
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))

from sketch_color_features import (  # noqa: E402
    emotion_colour, hue_from_valence, percentile, sat_from_arousal, spelling_seed, tint)


def test_percentile_counts_ties_as_half():
    v = np.sort(np.array([1.0, 2.0, 2.0, 3.0]))
    assert percentile(v, 0.5) == 0.0
    assert percentile(v, 2.0) == pytest.approx(0.5)
    assert percentile(v, 3.0) == pytest.approx(0.875)
    assert percentile(v, 9.0) == 1.0


def test_spelling_seed_depends_on_spelling_and_common_seed_only():
    assert spelling_seed("言う", 100) == spelling_seed("言う", 100)
    assert spelling_seed("言う", 100) != spelling_seed("云う", 100)
    assert spelling_seed("言う", 300) != spelling_seed("言う", 100)
    assert 0 <= spelling_seed("私", 100) < 2**31


def test_colour_scales_run_between_their_ends():
    assert hue_from_valence(0.0) == 220.0 and hue_from_valence(1.0) == 25.0
    assert sat_from_arousal(0.0) == pytest.approx(0.15) and sat_from_arousal(1.0) == pytest.approx(0.85)


def test_emotion_colour_takes_the_largest_z():
    z = {"joy": 0.5, "anger": 2.0, "sadness": -1.0, "fear": 0.0, "disgust": 1.9}
    assert emotion_colour(z) == (0.0, pytest.approx(0.05 + 0.8 * 2 / 3))
    assert emotion_colour({e: -1.0 for e in z})[1] == pytest.approx(0.05)
    assert emotion_colour({**{e: 0.0 for e in z}, "fear": 5.0}) == (280.0, pytest.approx(0.85))


def test_tint_keeps_the_lightness_of_the_grey():
    g = np.array([[0, 64], [128, 255]], dtype=np.uint8)
    assert np.array_equal(tint(g, None, 0.0)[..., 0], g)
    rgb = tint(g, 210.0, 0.7).astype(float) / 255
    light = np.array([[colorsys.rgb_to_hls(*rgb[i, j])[1] for j in range(2)] for i in range(2)])
    assert np.allclose(light, g / 255, atol=0.01)
