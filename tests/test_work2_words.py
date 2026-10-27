"""Tests of the selection and of the icosahedron of works/work2_words.py on counts written for the tests (no text of
any work, no lexicon)."""

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
sys.path.insert(0, str(ROOT / "checks"))

from work2_words import icosahedron, select, taper, view  # noqa: E402

# two authors, two works each; "a" is common to both works of X, "b" sits in one work of X, "c" is in both works of X
# but once in one of them, "d" is spread over X and Y
WORDS = {
    "x1": Counter({"a": 10, "b": 30, "c": 5, "d": 2}),
    "x2": Counter({"a": 10, "c": 1, "d": 2, "f": 30}),
    "y1": Counter({"d": 20, "e": 20}),
    "y2": Counter({"d": 20, "e": 20}),
}
AUTHOR = {"x1": "X", "x2": "X", "y1": "Y", "y2": "Y"}
TOKENS = {w: sum(c.values()) for w, c in WORDS.items()}


def words_of(r: dict, a: str) -> list[str]:
    return [d["word"] for d in r[a]["words"]]


def test_a_word_of_one_work_is_not_a_candidate():
    r = select(WORDS, AUTHOR, TOKENS, min_count=2, min_works=2, max_share=2 / 3, n_words=20)
    assert "b" not in words_of(r, "X")  # one work only
    assert "c" not in words_of(r, "X")  # once in x2
    assert "a" in words_of(r, "X")


def test_min_count_one_admits_a_word_seen_once_in_the_second_work():
    r = select(WORDS, AUTHOR, TOKENS, min_count=1, min_works=2, max_share=1.0, n_words=20)
    assert "c" in words_of(r, "X")


def test_share_is_computed_on_rates_not_on_counts():
    # equal rates in works of different length: share 1/2 although the counts are 2 and 20
    w = {"x1": Counter({"a": 2, "z": 8}), "x2": Counter({"a": 20, "z": 80}), "y1": Counter({"q": 50}),
         "y2": Counter({"q": 50})}
    tok = {k: sum(c.values()) for k, c in w.items()}
    r = select(w, AUTHOR, tok, min_count=2, min_works=2, max_share=0.5, n_words=20)
    assert {d["word"]: d["share"] for d in r["X"]["words"]}["a"] == 0.5


def test_only_positive_z_and_order_by_z():
    r = select(WORDS, AUTHOR, TOKENS, min_count=2, min_works=2, max_share=1.0, n_words=20)
    zs = [d["z"] for d in r["X"]["words"]]
    assert all(z > 0 for z in zs) and zs == sorted(zs, reverse=True)
    assert "d" not in words_of(r, "X")  # more frequent in Y


def test_candidates_counted_before_the_cut():
    r = select(WORDS, AUTHOR, TOKENS, min_count=1, min_works=2, max_share=1.0, n_words=1)
    assert len(r["X"]["words"]) == 1 and r["X"]["candidates"] >= 2


def test_icosahedron_has_twenty_outward_faces_of_the_edge():
    V, faces = icosahedron(50.0)
    assert len(faces) == 20 and len({frozenset(f) for f in faces}) == 20
    for a, b, c in faces:
        for i, j in ((a, b), (b, c), (c, a)):
            assert math.isclose(np.linalg.norm(V[i] - V[j]), 50.0)
        centre = (V[a] + V[b] + V[c]) / 3
        assert np.dot(np.cross(V[b] - V[a], V[c] - V[a]), centre) > 0
        assert V[c][2] >= max(V[a][2], V[b][2]) - 1e-9  # apex on top
    heights = [round(((V[a] + V[b] + V[c]) / 3)[2], 9) for a, b, c in faces]
    assert heights == sorted(heights, reverse=True)


def test_taper_is_zero_on_the_edges_and_one_inside():
    t = taper(400, 0.2)
    assert t[0, :].max() == 0.0  # the bottom edge y = 0
    assert t[int(0.25 * 400), 200] == 1.0  # near the centre (inradius 0.289)
    assert t[380, 10] == 0.0  # outside the triangle


def test_view_of_flat_faces_is_the_plain_solid():
    V, faces = icosahedron(50.0)
    flat = [np.zeros((50, 50)) for _ in faces]
    img = view(V, faces, flat, 50.0, [0.35, -0.85, 0.40], 120)
    assert img[0, 0] == 1.0 and img[60, 60] < 1.0  # background at the corner, the solid in the middle
    # a flat face is shaded uniformly: few distinct grey levels on the solid
    assert len(np.unique(np.round(img[img < 1.0], 6))) <= 20


def test_view_of_a_planar_slope_gives_the_expected_normal():
    # z = a x + b y (mm, x and y in domain units) on every face: the shade must be that of n - a' e1 - b' e2, with
    # a' = a / edge and b' = b / edge the slopes in mm per mm (catches a swapped axis, a wrong sign or unit)
    edge, n, a, b = 50.0, 200, 3.0, -2.0
    y, x = np.mgrid[0:n, 0:n] / n
    V, faces = icosahedron(edge)
    zs = [a * x + b * y for _ in faces]
    d = np.array([0.35, -0.85, 0.40])
    img = view(V, faces, zs, edge, d.tolist(), 400)
    d /= np.linalg.norm(d)
    right = np.cross([0.0, 0.0, 1.0], d)
    right /= np.linalg.norm(right)
    up = np.cross(d, right)
    light = 0.45 * d + 0.65 * up - 0.60 * right
    light /= np.linalg.norm(light)
    expected = set()
    for p, q, r in faces:
        e1 = (V[q] - V[p]) / edge
        e2 = (V[r] - V[p]) - np.dot(V[r] - V[p], e1) * e1
        e2 /= np.linalg.norm(e2)
        nf = np.cross(V[q] - V[p], V[r] - V[p])
        nf /= np.linalg.norm(nf)
        if np.dot(nf, d) <= 0:
            continue
        m = nf - a / edge * e1 - b / edge * e2
        m /= np.linalg.norm(m)
        expected.add(round(0.15 + 0.85 * (1 + m @ light) / 2, 4))
    seen = set(np.round(img[img < 1.0], 4))
    # interior pixels take exactly the expected shades (pixels next to an edge may sample across it)
    counts = {s: int(np.sum(np.round(img, 4) == s)) for s in expected}
    assert all(c > 100 for c in counts.values()), counts
    assert len(expected) >= 8 and expected <= seen
