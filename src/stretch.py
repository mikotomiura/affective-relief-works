"""Common linear stretch of the window means of work 2, and the rule that fixes its factor from a reference corpus.

The decisions (.steering/20261028-while-waiting-2/ decisions 3 and 4, made before any text of the work was computed):
- V' = k V and A' = k A with one k for both dimensions and for every author and work, then clipped to [-1, 1]. The
  zero of the calibrated map (map B of the Japanese text step) is not moved. The stretch is applied after the window
  means; gaps stay gaps. The unstretched trajectory is kept beside the stretched one.
- k = TARGET / Q, with Q the weighted QUANTILE of max(|V|, |A|) over the windows of the reference works that have a
  point (gaps left out); each work has the same total weight, shared equally by its windows. The reference works are
  works of other authors, chosen by a rule fixed before they were read (checks/conditions/reference_stretch.json).
- Weighted quantile: the values sorted ascending with their weights normalised to sum 1; Q is the smallest value
  whose cumulative weight reaches q (to 1e-12, so that exact fractions are not lost to rounding).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

from window import WindowPoint

TARGET = 0.9
QUANTILE = 0.99


def weighted_quantile(values: Sequence[float], weights: Sequence[float], q: float) -> float:
    if len(values) != len(weights) or not values:
        raise ValueError("values and weights must be non-empty and of the same length")
    if not 0.0 < q <= 1.0:
        raise ValueError(f"quantile {q}")
    if any(not math.isfinite(v) for v in values) or any(not (math.isfinite(w) and w > 0) for w in weights):
        raise ValueError("values must be finite and weights finite and positive")
    total = math.fsum(weights)
    acc = 0.0
    pairs = sorted(zip(values, weights))
    for v, w in pairs:
        acc += w / total
        if acc >= q - 1e-12:
            return v
    return pairs[-1][0]


def reference_values(works: Sequence[Sequence[WindowPoint]]) -> tuple[list[float], list[float]]:
    """max(|V|, |A|) of every window with a point and its weight (1 / windows with a point, per work)."""
    vals: list[float] = []
    wts: list[float] = []
    for pts in works:
        have = [p for p in pts if not p.gap]
        if not have:
            raise ValueError("a reference work has no window with a point")
        for p in have:
            if not (math.isfinite(p.V) and math.isfinite(p.A)):
                raise ValueError(f"non-finite window point {p}")
            vals.append(max(abs(p.V), abs(p.A)))
            wts.append(1.0 / len(have))
    return vals, wts


def factor(works: Sequence[Sequence[WindowPoint]], target: float = TARGET, q: float = QUANTILE) -> tuple[float, float]:
    """(k, Q) from the reference works."""
    if not (math.isfinite(target) and target > 0):
        raise ValueError(f"target {target}")
    Q = weighted_quantile(*reference_values(works), q)
    if not Q > 0:
        raise ValueError("the quantile is zero; no stretch can be fixed")
    return target / Q, Q


@dataclass(frozen=True)
class Stretched:
    points: list[WindowPoint]
    clipped: int  # windows with a point of which V or A was clipped
    with_point: int


def stretch(points: Sequence[WindowPoint], k: float) -> Stretched:
    if isinstance(k, bool) or not isinstance(k, (int, float)) or not (math.isfinite(k) and k > 0):
        raise ValueError(f"factor {k!r}")
    out, clipped, have = [], 0, 0
    for p in points:
        if p.gap:
            out.append(p)
            continue
        have += 1
        if not (math.isfinite(p.V) and math.isfinite(p.A)):
            raise ValueError(f"non-finite window point {p}")  # min/max would turn NaN into a bound
        v, a = k * p.V, k * p.A
        cv, ca = min(1.0, max(-1.0, v)), min(1.0, max(-1.0, a))
        clipped += (cv != v) or (ca != a)
        out.append(replace(p, V=cv, A=ca))
    return Stretched(out, clipped, have)


def clip_rate(works: Sequence[Sequence[WindowPoint]], k: float) -> float:
    """Weighted share of the windows with a point that are clipped under k (each work the same total weight)."""
    num = 0.0
    for pts in works:
        s = stretch(pts, k)
        num += s.clipped / s.with_point
    return num / len(works)
