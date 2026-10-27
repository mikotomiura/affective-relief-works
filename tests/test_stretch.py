import math

import numpy as np
import pytest

import stretch
from window import Sentence, WindowPoint, trajectory


def series(V, A, matched=None):
    m = [True] * len(V) if matched is None else matched
    return [Sentence(float(v), float(a), bool(x)) for v, a, x in zip(V, A, m)]


def test_weighted_quantile():
    assert stretch.weighted_quantile([3, 1, 2], [1, 1, 1], 0.5) == 2
    assert stretch.weighted_quantile([3, 1, 2], [1, 1, 1], 1.0) == 3
    assert stretch.weighted_quantile([1, 2], [1, 3], 0.25) == 1  # exactly reached
    assert stretch.weighted_quantile([1, 2], [1, 3], 0.26) == 2
    assert stretch.weighted_quantile([0.1 * i for i in range(1, 11)], [1] * 10, 0.3) == pytest.approx(0.3)
    for bad in (([], [], 0.5), ([1], [1, 2], 0.5), ([1], [0], 0.5), ([1], [1], 0.0), ([math.nan], [1], 0.5)):
        with pytest.raises(ValueError):
            stretch.weighted_quantile(*bad)


def test_each_work_has_the_same_weight():
    short = trajectory(series([0.1] * 50, [0.0] * 50))  # 11 windows
    long = trajectory(series([0.5] * 1000, [0.0] * 1000))  # 961 windows
    v, w = stretch.reference_values([short, long])
    assert math.fsum(w) == pytest.approx(2.0)
    assert stretch.weighted_quantile(v, w, 0.5) == pytest.approx(0.1)  # unweighted, the long work would decide
    assert stretch.weighted_quantile(v, w, 0.51) == pytest.approx(0.5)


def test_factor_uses_the_larger_of_the_two_dimensions():
    t = trajectory(series([0.1] * 60, [-0.3] * 60))
    k, Q = stretch.factor([t])
    assert Q == pytest.approx(0.3) and k == pytest.approx(3.0)


def test_gaps_are_left_out_of_the_reference_and_kept_by_the_stretch():
    m = [True] * 60 + [False] * 60 + [True] * 60
    t = trajectory(series([0.2] * 180, [0.1] * 180, m))
    assert any(p.gap for p in t)
    v, _ = stretch.reference_values([t])
    assert len(v) == sum(not p.gap for p in t)
    s = stretch.stretch(t, 2.0)
    assert [p.gap for p in s.points] == [p.gap for p in t]
    assert s.with_point == len(v)
    with pytest.raises(ValueError):
        stretch.reference_values([trajectory(series([0.2] * 40, [0.1] * 40, [False] * 40))])


@pytest.mark.parametrize("k", [0.0, -1.0, math.inf, math.nan, True])
def test_bad_factor(k):
    with pytest.raises(ValueError):
        stretch.stretch([WindowPoint(0, 0.5, 40, 0.1, 0.1)], k)


def synthetic(kind: str, n: int = 600, seed: int = 1) -> list[Sentence]:
    rng = np.random.default_rng(seed)
    i = np.arange(n)
    if kind == "white":
        V, A = rng.normal(0, 0.3, n), rng.normal(0.05, 0.2, n)
    elif kind == "correlated":  # AR(1), phi 0.9
        V, A = np.zeros(n), np.zeros(n)
        for t in range(1, n):
            V[t] = 0.9 * V[t - 1] + rng.normal(0, 0.1)
            A[t] = 0.9 * A[t - 1] + rng.normal(0, 0.1)
    elif kind == "step":
        V, A = np.where(i < n // 2, -0.2, 0.2), np.where(i < n // 2, 0.1, -0.1)
    elif kind == "sine":
        V, A = 0.3 * np.sin(2 * np.pi * i / 200), 0.2 * np.cos(2 * np.pi * i / 200)
    else:
        raise ValueError(kind)
    matched = rng.random(n) > 0.3  # about 30 % of the sentences have no matched word
    matched[200:260] = False  # a run long enough to cut the line
    return series(np.clip(V, -1, 1), np.clip(A, -1, 1), matched)


@pytest.mark.parametrize("kind", ["white", "correlated", "step", "sine"])
def test_stretch_on_synthetic_series(kind):
    t = trajectory(synthetic(kind))
    k, Q = stretch.factor([t])
    s = stretch.stretch(t, k)
    assert any(p.gap for p in t)
    clipped = 0
    for a, b in zip(t, s.points):
        assert a.gap == b.gap and a.n_matched == b.n_matched and a.position == b.position
        if a.gap:
            continue
        assert b.V == min(1.0, max(-1.0, k * a.V)) and b.A == min(1.0, max(-1.0, k * a.A))
        assert math.copysign(1, b.V) == math.copysign(1, a.V) or a.V == 0
        clipped += abs(k * a.V) > 1 or abs(k * a.A) > 1
    assert s.clipped == clipped
    # on its own reference, at most 1 % of the windows go past 0.9 / 0.9 = 1 and are clipped
    assert stretch.clip_rate([t], k) <= 0.01 + 1e-12
    # the window mean of white noise is narrower than the sentence points (the shrinkage the stretch answers)
    if kind == "white":
        sent = [abs(x.V) for x in synthetic(kind) if x.matched]
        assert np.percentile([abs(p.V) for p in t if not p.gap], 99) < 0.5 * np.percentile(sent, 99)


def test_stretch_preserves_order_below_the_clip():
    t = trajectory(synthetic("sine"))
    s = stretch.stretch(t, 1.5)
    pairs = [(a.V, b.V) for a, b in zip(t, s.points) if not a.gap and abs(1.5 * a.V) < 1]
    xs = sorted(pairs)
    assert all(b1 <= b2 for (_, b1), (_, b2) in zip(xs, xs[1:]))


def test_clip_rate_weights_works_equally():
    a = trajectory(series([0.95] * 50, [0.0] * 50))  # every window clipped under k = 2
    b = trajectory(series([0.1] * 500, [0.0] * 500))  # none clipped
    assert stretch.clip_rate([a, b], 2.0) == pytest.approx(0.5)


def test_each_dimension_is_clipped_and_counted():
    s = stretch.stretch([WindowPoint(0, 0.5, 40, 0.1, 0.8), WindowPoint(1, 0.6, 40, -0.7, 0.1)], 2.0)
    assert (s.points[0].V, s.points[0].A) == (pytest.approx(0.2), 1.0)
    assert (s.points[1].V, s.points[1].A) == (-1.0, pytest.approx(0.2))
    assert s.clipped == 2
