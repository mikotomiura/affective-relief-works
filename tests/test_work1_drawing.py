"""Tests of the shading and the placing of tiles of works/work1_drawing.py on surfaces written for the tests (no
sentence, no lexicon)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))
sys.path.insert(0, str(ROOT / "src"))

import work1  # noqa: E402
from work1_drawing import block_mean, lambert, stretch, tile_place, u8  # noqa: E402


def bump(n: int = 120, L: float = 50.0) -> np.ndarray:
    y, x = np.mgrid[0:n, 0:n] / n * L
    c = L * (n - 1) / (2 * n)  # the centre of the grid, so that the bump is symmetric under a half turn of the array
    return 0.6 * np.exp(-((x - c) ** 2 + (y - c) ** 2) / (2 * 6.0**2))


def test_flat_surface_is_the_sine_of_the_elevation():
    g = lambert(np.zeros((10, 10)), 50.0, 135.0, 45.0)
    assert np.allclose(g, math.sin(math.radians(45.0)))


def test_light_from_the_upper_left_lights_the_upper_left_of_a_bump_in_the_figure():
    fig_ul = np.flipud(lambert(bump(), 50.0, 135.0, 45.0))  # rows from the top, as drawn
    fig_lr = np.flipud(lambert(bump(), 50.0, 315.0, 45.0))
    h = fig_ul.shape[0] // 2
    assert fig_ul[:h, :h].mean() > fig_ul[h:, h:].mean()
    assert fig_lr[h:, h:].mean() > fig_lr[:h, :h].mean()
    assert np.allclose(fig_ul, np.rot90(fig_lr, 2), atol=1e-12)  # the two lights are the same picture turned


def test_steep_slope_away_from_the_light_is_black():
    z = np.tile(np.arange(10) * 50.0, (10, 1))  # spacing 5 mm, rising 50 mm per column: slope 10 towards +x
    g = lambert(z, 50.0, 0.0, 10.0)  # the light comes low from +x; the face looks towards -x, away from it
    assert np.all(g == 0.0)


def test_block_mean_and_stretch():
    x = np.arange(16.0).reshape(4, 4)
    assert np.allclose(block_mean(x, 2), [[2.5, 4.5], [10.5, 12.5]])
    with pytest.raises(ValueError):
        block_mean(np.zeros((5, 5)), 2)
    s = stretch(np.array([[0.2, 0.4], [0.3, 0.6]]))
    assert s.min() == 0.0 and s.max() == 1.0
    with pytest.raises(ValueError):
        stretch(np.full((2, 2), 0.3))


def test_tile_place_matches_the_figure_of_work1():
    fig = {"width_mm": 177.8, "gap_mm": 1.0, "dpi": 300}
    lay = work1.layout(104, fig)
    tiles = [np.full((lay["tile_px"], lay["tile_px"]), (i % 7) / 7) for i in range(104)]
    img = np.asarray(work1.figure(tiles, lay, fig))
    px = lay["tile_px"]
    for i in (0, 1, 10, 11, 57, 103):
        y, x = tile_place(i, lay, fig)
        assert np.array_equal(img[y : y + px, x : x + px], u8(tiles[i]))


def test_u8_puts_the_origin_at_the_lower_left():
    g = np.zeros((4, 4))
    g[0, 0] = 1.0  # row 0 is y = 0, the bottom of the tile
    img = u8(g)
    assert img[3, 0] == 255 and img[0, 0] == 0
