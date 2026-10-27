"""Tests of the regions of works/work2_icosa_stripes.py against view_k() of works/work2_icosa_v2.py, on heights
written for the tests (no word, no lexicon)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
sys.path.insert(0, str(ROOT / "src"))

from work2_icosa_stripes import blocks, hits, region_rays, single_face  # noqa: E402
from work2_icosa_v2 import gradients, shade, view_k  # noqa: E402
from work2_words import icosahedron  # noqa: E402

DIRECTION = [0.35, -0.85, 0.40]


def bumpy(n: int, count: int) -> list[np.ndarray]:
    y, x = np.mgrid[0:n, 0:n] / n
    return [0.3 * np.sin(2 * np.pi * (3 + i) * x) * np.cos(2 * np.pi * 5 * y) for i in range(count)]


def test_a_region_is_the_crop_of_the_whole_view():
    V, faces = icosahedron(50.0)
    grads = gradients(bumpy(100, len(faces)), 50.0)
    whole = view_k(V, faces, grads, 50.0, DIRECTION, 60, 3, 60)
    U, W = region_rays(V, 60, 3, [20, 35], [10, 50])
    part = blocks(shade(V, faces, grads, 50.0, 100, DIRECTION, U, W), 3)
    assert np.array_equal(part, whole[20:35, 10:50])


def test_faces_unused_in_a_region_need_no_heights():
    V, faces = icosahedron(50.0)
    zs = bumpy(100, len(faces))
    U, W = region_rays(V, 60, 2, [25, 35], [25, 35])
    face, _, _ = hits(V, faces, 50.0, DIRECTION, U, W)
    needed = {int(f) for f in np.unique(face) if f >= 0}
    assert 0 < len(needed) < len(faces)
    full = gradients(zs, 50.0)
    small = np.zeros((2, 2))
    sparse = [g if i in needed else (small, small) for i, g in enumerate(full)]
    assert np.array_equal(shade(V, faces, full, 50.0, 100, DIRECTION, U, W),
                          shade(V, faces, sparse, 50.0, 100, DIRECTION, U, W))


def test_hits_give_the_coordinates_of_the_face():
    # a ray hitting a face lands inside its triangle (0,0), (1,0), (1/2, sqrt3/2)
    V, faces = icosahedron(50.0)
    U, W = region_rays(V, 40, 2, [0, 40], [0, 40])
    face, qx, qy = hits(V, faces, 50.0, DIRECTION, U, W)
    m = face >= 0
    s3 = np.sqrt(3)
    assert m.sum() > 1000 and (~m).sum() > 100
    tol = 1e-9
    assert np.all(qy[m] >= -tol) and np.all(s3 * qx[m] - qy[m] >= -tol) and np.all(s3 * (1 - qx[m]) - qy[m] >= -tol)


def test_single_face_leaves_out_outline_and_ridges():
    face = np.full((12, 12), -1)
    face[0:12, 0:6] = 0
    face[0:12, 6:12] = 1
    face[0:2, :] = -1  # a missed row at the top
    m = single_face(face, 2)  # 6 x 6 pixels
    assert not m[0].any()  # the pixels next to the missed row (3 x 3 neighbourhood) and the row itself
    assert not m[:, 2].any() and not m[:, 3].any()  # the ridge between faces 0 and 1
    assert m[3, 1] and m[3, 4]
