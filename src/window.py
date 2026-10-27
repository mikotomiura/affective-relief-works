"""Window from the sentence points of one work to its trajectory, written for work 2 (one portrait per author).

The rule (decided before any text of the work was computed, .steering/20261028-while-waiting/ decision 4):
- Unit: sentences. The window holds W consecutive sentences of one work (W = 40 for work 2); it moves by one sentence,
  so a work of n sentences gives n - W + 1 windows. No window is shorter than W: the ends of a work are not padded.
  A work shorter than W is an error (stopped, not given zero windows): the works are chosen with at least 200
  sentences, so a shorter one means a wrong input.
- Works are never joined: each work has its own trajectory (an author's works are separate lines).
- Position: the centre of the window as a fraction of the work, (start + W / 2) / n, in (0, 1). In the portrait every
  work is given the same length, so a work's weight is the length it is drawn over, not its number of sentences.
- Mean: the points of the matched sentences in the window, each with weight 1 (a sentence with no matched word has
  no point and is left out; it is not counted as (0, 0)). The number of matched sentences is kept with every window.
- Gap: if fewer than MIN_MATCHED_FRACTION of the W sentences are matched (fewer than 20 of 40), the window has no
  point and the line is cut there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

W_WORK2 = 40
MIN_MATCHED_FRACTION = 0.5


@dataclass(frozen=True)
class Sentence:
    V: float
    A: float
    matched: bool  # False if no word of the sentence was matched (Z = 0); V and A are then not used


@dataclass(frozen=True)
class WindowPoint:
    start: int  # index of the first sentence of the window
    position: float  # centre of the window as a fraction of the work
    n_matched: int
    V: float | None  # None for a gap
    A: float | None

    @property
    def gap(self) -> bool:
        return self.V is None


def min_matched(w: int) -> int:
    """Smallest number of matched sentences for a window of w sentences to have a point."""
    return math.ceil(MIN_MATCHED_FRACTION * w)


def trajectory(sentences: Sequence[Sentence], w: int = W_WORK2) -> list[WindowPoint]:
    """Windows of one work, in order."""
    n = len(sentences)
    if not isinstance(w, int) or isinstance(w, bool) or w < 1:
        raise ValueError(f"window of {w} sentences")
    if n < w:
        raise ValueError(f"a work of {n} sentences is shorter than the window of {w}")
    for s in sentences:
        if s.matched and not (math.isfinite(s.V) and math.isfinite(s.A)):
            raise ValueError(f"non-finite point {s}")
    need = min_matched(w)
    out = []
    for i in range(n - w + 1):
        pts = [s for s in sentences[i : i + w] if s.matched]
        pos = (i + w / 2) / n
        if len(pts) < need:
            out.append(WindowPoint(i, pos, len(pts), None, None))
        else:
            out.append(WindowPoint(i, pos, len(pts), sum(s.V for s in pts) / len(pts), sum(s.A for s in pts) / len(pts)))
    return out


def segments(points: Sequence[WindowPoint]) -> list[list[WindowPoint]]:
    """Runs of consecutive windows that have a point: the pieces of line to draw (a gap cuts the line)."""
    runs: list[list[WindowPoint]] = []
    cur: list[WindowPoint] = []
    for p in points:
        if p.gap:
            if cur:
                runs.append(cur)
            cur = []
        else:
            cur.append(p)
    if cur:
        runs.append(cur)
    return runs


def author_trajectories(works: Sequence[Sequence[Sentence]], w: int = W_WORK2) -> list[list[WindowPoint]]:
    """One trajectory per work, in the given order; works are not joined."""
    return [trajectory(s, w) for s in works]
