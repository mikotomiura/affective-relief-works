"""Tests of the supersampled views of works/work2_icosa_v2.py against view() of works/work2_words.py, on heights
written for the tests (no word, no lexicon)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
sys.path.insert(0, str(ROOT / "src"))

from work2_icosa_v2 import gradients, grid, view_k  # noqa: E402
from work2_words import icosahedron, view  # noqa: E402

DIRECTION = [0.35, -0.85, 0.40]


def bumpy(n: int, faces: list) -> list[np.ndarray]:
    y, x = np.mgrid[0:n, 0:n] / n
    return [0.3 * np.sin(2 * np.pi * (3 + i) * x) * np.cos(2 * np.pi * 5 * y) for i in range(len(faces))]


def test_k1_equals_the_old_view():
    V, faces = icosahedron(50.0)
    zs = bumpy(120, faces)
    old = view(V, faces, zs, 50.0, DIRECTION, 90)
    new = view_k(V, faces, gradients(zs, 50.0), 50.0, DIRECTION, 90, 1, 7)  # 7 rows per chunk: the chunks do not divide 90
    assert np.array_equal(old, new)


def test_chunks_do_not_change_the_image():
    V, faces = icosahedron(50.0)
    zs = bumpy(120, faces)
    a = view_k(V, faces, gradients(zs, 50.0), 50.0, DIRECTION, 60, 3, 1)
    b = view_k(V, faces, gradients(zs, 50.0), 50.0, DIRECTION, 60, 3, 60)
    assert np.array_equal(a, b)


def test_supersampling_keeps_a_flat_face_and_averages_the_outline():
    V, faces = icosahedron(50.0)
    flat = [np.zeros((40, 40)) for _ in faces]
    one = view_k(V, faces, gradients(flat, 50.0), 50.0, DIRECTION, 80, 1, 80)
    four = view_k(V, faces, gradients(flat, 50.0), 50.0, DIRECTION, 80, 4, 80)
    # inside a flat face (a 5 x 5 neighbourhood of one shade) every sub-position has that shade
    from scipy import ndimage  # noqa: E402

    uniform = (ndimage.maximum_filter(one, 5) == ndimage.minimum_filter(one, 5)) & (one < 1.0)
    assert uniform.sum() > 500
    assert np.allclose(four[uniform], one[uniform])
    assert four[0, 0] == 1.0  # background
    # the outline gets intermediate values that one ray per pixel cannot give
    assert np.sum((four > one.min()) & (four < 1.0) & (one == 1.0)) > 0


def test_supersampling_is_the_mean_of_finer_rays():
    # with k = 2, a pixel is the mean of the four sub-positions: build them by hand from a view with k = 1
    V, faces = icosahedron(50.0)
    zs = bumpy(80, faces)
    px = 20
    ext = 1.08 * np.linalg.norm(V[0])
    delta = 2 * ext / (px - 1)
    two = view_k(V, faces, gradients(zs, 50.0), 50.0, DIRECTION, px, 2, px)
    from work2_icosa_v2 import shade  # noqa: E402

    grads = [tuple(np.gradient(z, 50.0 / 80)) for z in zs]
    i, j = 9, 11
    cu, cw = -ext + j * delta, ext - i * delta
    pts = [(cu + su, cw + sw) for su in (-delta / 4, delta / 4) for sw in (-delta / 4, delta / 4)]
    U = np.array([[p[0] for p in pts]])
    W = np.array([[p[1] for p in pts]])
    expected = shade(V, faces, grads, 50.0, 80, DIRECTION, U, W).mean()
    assert expected < 1.0  # the pixel is on the solid
    assert np.isclose(two[i, j], expected)


def test_grid_places_images_with_gaps():
    a = np.zeros((2, 3), np.uint8)
    g = grid([[a, a], [a]], gap=1)
    assert g.shape == (5, 7)
    assert g[2, 0] == 255 and g[0, 3] == 255 and g[3, 4] == 255 and g[3, 2] == 0
