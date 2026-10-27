"""Tests of the window on synthetic sequences (no text, no lexicon)."""

from __future__ import annotations

import math
import random

import pytest

from window import Sentence, author_trajectories, min_matched, segments, trajectory


def seq(values, matched=None):
    m = matched or [True] * len(values)
    return [Sentence(v, -v, ok) for v, ok in zip(values, m)]


def test_constant_sequence_gives_constant_windows_and_positions():
    t = trajectory(seq([0.3] * 100), w=40)
    assert len(t) == 61
    assert all(p.V == pytest.approx(0.3) and p.A == pytest.approx(-0.3) and p.n_matched == 40 for p in t)
    assert t[0].position == pytest.approx(20 / 100) and t[-1].position == pytest.approx(80 / 100)


def test_step_becomes_a_ramp_over_one_window():
    t = trajectory(seq([0.0] * 60 + [1.0] * 60), w=40)
    vals = [p.V for p in t]
    assert vals[20] == pytest.approx(0.0) and vals[60] == pytest.approx(1.0)
    assert vals[40] == pytest.approx(0.5)
    assert all(b >= a for a, b in zip(vals, vals[1:]))


@pytest.mark.parametrize("period", [30.0, 80.0, 160.0])
def test_sinusoid_is_scaled_by_the_boxcar_gain_with_its_sign(period):
    w = 40
    vals = [0.8 * math.sin(2 * math.pi * k / period) for k in range(400)]
    t = trajectory(seq(vals), w=w)
    gain = math.sin(math.pi * w / period) / (w * math.sin(math.pi / period))
    for p in t:
        centre = p.start + (w - 1) / 2
        assert p.V == pytest.approx(gain * 0.8 * math.sin(2 * math.pi * centre / period), abs=1e-12)
    if period == 30.0:
        assert gain < 0  # shorter than the window: the wave is reversed, as the decision records
    if period == 80.0:
        assert gain == pytest.approx(0.6366, abs=1e-3)


def test_white_noise_is_reduced_about_one_over_root_w():
    rng = random.Random(7)
    vals = [rng.gauss(0, 1) for _ in range(20000)]
    t = trajectory(seq(vals), w=40)
    v = [p.V for p in t[::40]]  # non-overlapping windows
    sd = math.sqrt(sum(x * x for x in v) / len(v))
    assert 0.8 / math.sqrt(40) < sd < 1.2 / math.sqrt(40)


def test_unmatched_sentences_are_left_out_not_counted_as_zero():
    matched = [True] * 30 + [False] * 10
    t = trajectory([Sentence(0.6, -0.4, True) if ok else Sentence(9.0, 9.0, False) for ok in matched], w=40)
    assert t[0].V == pytest.approx(0.6) and t[0].A == pytest.approx(-0.4) and t[0].n_matched == 30


def test_gap_below_half_matched():
    w = 40
    assert min_matched(w) == 20
    for k, gap in ((19, True), (20, False)):
        matched = [True] * k + [False] * (w - k)
        p = trajectory([Sentence(0.5, 0.5, ok) for ok in matched], w=w)[0]
        assert p.gap is gap and p.n_matched == k


def test_segments_cut_at_gaps():
    matched = [True] * 60 + [False] * 60 + [True] * 60
    t = trajectory([Sentence(0.1, 0.1, ok) for ok in matched], w=40)
    runs = segments(t)
    starts = [[p.start for p in r] for r in runs]
    assert starts == [[i for i in range(len(t)) if not t[i].gap and i < 60], [i for i in range(len(t)) if not t[i].gap and i >= 60]]
    assert starts[0][0] == 0 and starts[1][-1] == len(t) - 1 and all(not p.gap for r in runs for p in r)


def test_works_are_not_joined():
    a, b = seq([1.0] * 50), seq([-1.0] * 70)
    ta, tb = author_trajectories([a, b], w=40)
    assert len(ta) == 11 and len(tb) == 31
    assert all(p.V == pytest.approx(1.0) for p in ta) and all(p.V == pytest.approx(-1.0) for p in tb)


def test_short_work_and_bad_values_fail():
    with pytest.raises(ValueError, match="shorter"):
        trajectory(seq([0.0] * 39), w=40)
    with pytest.raises(ValueError, match="window of 0"):
        trajectory(seq([0.0] * 5), w=0)
    with pytest.raises(ValueError, match="non-finite"):
        trajectory(seq([0.0] * 39 + [float("nan")]), w=40)


def test_non_finite_value_of_an_unmatched_sentence_is_not_used():
    s = [Sentence(0.2, 0.2, True)] * 39 + [Sentence(float("nan"), 0.0, False)]
    assert trajectory(s, w=40)[0].V == pytest.approx(0.2)


def test_gap_threshold_rounds_up_for_an_odd_window():
    assert min_matched(5) == 3
    for k, gap in ((2, True), (3, False)):
        p = trajectory([Sentence(0.5, 0.5, i < k) for i in range(5)], w=5)[0]
        assert p.gap is gap


def test_work_as_long_as_the_window_gives_one_window_at_the_centre():
    t = trajectory(seq([0.1] * 40), w=40)
    assert len(t) == 1 and t[0].start == 0 and t[0].position == pytest.approx(0.5)


def test_window_of_one_and_independent_components():
    s = [Sentence(0.1 * k, -0.3 * k, True) for k in range(4)]
    t = trajectory(s, w=1)
    assert [(p.V, p.A) for p in t] == [(x.V, x.A) for x in s]
    t2 = trajectory(s, w=2)
    assert [(round(p.V, 12), round(p.A, 12)) for p in t2] == [(0.05, -0.15), (0.15, -0.45), (0.25, -0.75)]


def test_window_length_must_be_an_integer():
    with pytest.raises(ValueError, match="window of 40.0"):
        trajectory(seq([0.0] * 50), w=40.0)
