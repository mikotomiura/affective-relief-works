import math

import numpy as np
import pytest

import relief
from atact.field import procedural_field_separable
from atact.transform import t_gamma_eps

D = relief.Dimensions()


def test_eps_and_grid_of_the_decisions():
    assert relief.eps_for(D) == pytest.approx(0.3529079159926097, rel=1e-15)
    assert relief.grid_intervals(D) == 250
    with pytest.raises(ValueError):
        relief.grid_intervals(relief.Dimensions(L_mm=50.1))
    for bad in (0, -1.0, float("nan"), True):
        with pytest.raises(ValueError):
            relief.Dimensions(base_mm=bad)


def test_closed_field_inner_block_is_the_generator_call():
    F = relief.closed_field(5.3, 16, 42)
    assert F.shape == (17, 17)
    G = procedural_field_separable(5.3, 16, 42, octaves=4, normalize="ampsum")
    assert np.array_equal(F[:16, :16], G)


def test_closed_field_edges_are_the_same_points_on_a_finer_grid():
    # x = 1 and y = 1 on the grid of n intervals are the same coordinates as on the grid of 2n intervals
    a, b = relief.closed_field(6.1, 12, 7), relief.closed_field(6.1, 24, 7)
    assert np.array_equal(a, b[::2, ::2])
    assert not np.array_equal(a[:, -1], a[:, 0])  # the edge is computed, not copied


def test_closed_field_is_not_periodic_copy_and_depends_on_seed():
    assert not np.array_equal(relief.closed_field(4.0, 8, 1), relief.closed_field(4.0, 8, 2))
    with pytest.raises(ValueError):
        relief.closed_field(4.0, 0, 1)


def test_smoothed_heights_stay_in_the_span():
    for f in (3.0, 5.5, 7.0):
        F = relief.closed_field(f, 64, 42)
        assert np.abs(F).max() < 1.0
        for g in (0.9, 1.7, 2.6):
            assert np.abs(t_gamma_eps(F, g, relief.eps_for(D))).max() <= 1.0


@pytest.mark.parametrize("V,A", [(-1.0, -1.0), (1.0, 1.0), (0.3, -0.7), (-1.0, 1.0)])
def test_scales(V, A):
    t = relief.tile(V, A, lambda f: relief.closed_field(f, 32, 42), relief.eps_for(D))
    zc = relief.scaled(t.H_eps, "common", D)
    assert np.abs(zc).max() <= D.D_mm / 2 * t.s / D.s_max + 1e-12
    zm = relief.scaled(t.H_eps, "minmax", D)
    assert zm.min() == 0.0 and zm.max() == pytest.approx(D.D_mm)
    for scale in relief.SCALES:
        top = relief.top_surface(t.H_eps, scale, D)
        assert top.min() >= D.base_mm - 1e-12 and top.max() <= D.base_mm + D.D_mm + 1e-12
        g = relief.grey(t.H, scale, D)
        assert 0.0 <= g.min() and g.max() <= 1.0
    with pytest.raises(ValueError):
        relief.scaled(t.H, "other", D)
    with pytest.raises(ValueError):
        relief.scaled(np.zeros((3, 3)), "minmax", D)


def test_common_scale_keeps_the_depth_difference_and_minmax_removes_it():
    lo = relief.tile(1.0, 0.0, lambda f: relief.closed_field(f, 32, 42), 0.35)  # s smallest
    hi = relief.tile(-1.0, 0.0, lambda f: relief.closed_field(f, 32, 42), 0.35)  # s largest
    span = lambda H, sc: np.ptp(relief.scaled(H, sc, D))  # noqa: E731
    assert span(hi.H_eps, "common") > 2 * span(lo.H_eps, "common")
    assert span(hi.H_eps, "minmax") == pytest.approx(span(lo.H_eps, "minmax"))


def ramp(n=8):
    x = np.arange(n + 1) / n
    return 2.0 + np.add.outer(0.1 * x, x)  # rows y, columns x: rises with x ten times faster than with y


def test_solid_is_closed_and_oriented():
    top = ramp()
    tris = relief.solid(top, 10.0)
    n = top.shape[0] - 1
    assert len(tris) == 2 * n * n + 2 * 4 * n + 4 * n
    ec = relief.edge_check(tris)
    assert ec["not_two"] == 0 and ec["same_direction"] == 0 and ec["edges"] > 0
    nm = relief.normals(tris)
    assert np.all(nm[: 2 * n * n, 2] > 0)  # top up
    assert np.all(nm[-4 * n :, 2] < 0)  # bottom down
    # volume: the prisms under the top triangles
    h = 10.0 / n
    exp = sum(h * h / 2 * t[:, 2].mean() for t in tris[: 2 * n * n])
    assert relief.signed_volume(tris) == pytest.approx(exp, rel=1e-12)


def test_solid_flipped_winding_is_caught():
    tris = relief.solid(ramp(), 10.0)
    bad = tris.copy()
    bad[0] = bad[0][::-1]
    assert relief.edge_check(bad)["same_direction"] > 0


def test_solid_orientation_x_is_columns():
    top = ramp()
    tris = relief.solid(top, 10.0)
    v = tris.reshape(-1, 3)
    hi = v[v[:, 2] == v[:, 2].max()]
    assert np.all(hi[:, 0] == 10.0) and np.all(hi[:, 1] == 10.0)
    # the highest point along y = 0 is at x = L (columns are x)
    south = v[(v[:, 1] == 0.0) & (v[:, 2] > 0)]
    assert south[np.argmax(south[:, 2]), 0] == 10.0
    # height at x = L i / n, y = L j / n is top[j, i]
    pt = v[(v[:, 0] == 10.0 * 3 / 8) & (v[:, 1] == 10.0 * 5 / 8) & (v[:, 2] > 0)]
    assert np.all(pt[:, 2] == top[5, 3])


def test_solid_rejects_bad_tops():
    for bad in (np.ones((3, 4)), np.ones((1, 1)), np.zeros((3, 3)), np.full((3, 3), np.nan)):
        with pytest.raises(ValueError):
            relief.solid(bad, 1.0)


def test_stl_round_trip(tmp_path):
    tris = relief.solid(ramp(), 10.0)
    p = tmp_path / "t.stl"
    relief.write_stl(p, tris, "test tile")
    assert p.stat().st_size == 84 + 50 * len(tris)
    nm, back = relief.read_stl(p)
    assert np.allclose(back, tris, atol=1e-6)
    assert np.allclose(nm, relief.normals(tris), atol=1e-6)
    assert relief.edge_check(back)["not_two"] == 0 and relief.edge_check(back)["same_direction"] == 0
    with pytest.raises(FileExistsError):
        relief.write_stl(p, tris, "again")
    with pytest.raises(ValueError):
        relief.write_stl(tmp_path / "s.stl", tris, "solid name")
    (tmp_path / "short.stl").write_bytes(p.read_bytes()[:-1])
    with pytest.raises(ValueError):
        relief.read_stl(tmp_path / "short.stl")


def test_stl_independent_reader(tmp_path):
    trimesh = pytest.importorskip("trimesh")  # not a dependency of the project: uv run --with trimesh pytest
    t = relief.tile(0.2, 0.6, lambda f: relief.closed_field(f, 40, 42), relief.eps_for(D))
    top = relief.top_surface(t.H_eps, "common", D)
    p = tmp_path / "t.stl"
    relief.write_stl(p, relief.solid(top, D.L_mm), "independent check")
    m = trimesh.load(p, file_type="stl")
    assert m.is_watertight and m.is_winding_consistent and m.volume > 0
    assert np.allclose(m.bounds, [[0, 0, 0], [D.L_mm, D.L_mm, top.max()]], atol=1e-5)
    assert math.isclose(float(m.volume), relief.signed_volume(relief.solid(top, D.L_mm)), rel_tol=1e-5)


def test_closed_field_edge_by_an_independent_offset():
    # x = (n - 1)/n + 1/n and y = (n - 1)/n + 1/n reach the edge by another route (equal to rounding)
    n, f = 20, 6.3
    F = relief.closed_field(f, n, 42)
    shifted = procedural_field_separable(f, n, 42, octaves=4, x0=1 / n, y0=1 / n, normalize="ampsum")
    assert np.allclose(F[1:, n], shifted[:, n - 1], atol=1e-12)
    assert np.allclose(F[n, 1:], shifted[n - 1, :], atol=1e-12)
