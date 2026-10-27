"""Supplement to the STL decisions: a finer grid (0.1 mm) and eps from the analytic bound, at three points of the rule.

Usage:  uv run python checks/grid_eps_supplement.py      (refuses to overwrite)
Conditions: checks/conditions/grid_eps_supplement.json (committed before this script was run)
Output: results/grid_eps_supplement.json and output/grid_eps_supplement/ (STL, outside the repository)
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.pycache_prefix = tempfile.mkdtemp(prefix="pycache-")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "works"))

import numpy as np  # noqa: E402

import _common as cm  # noqa: E402
import relief  # noqa: E402

COND = ROOT / "checks" / "conditions" / "grid_eps_supplement.json"
OUT = ROOT / "results" / "grid_eps_supplement.json"
STL_DIR = ROOT / "output" / "grid_eps_supplement"
SCRIPT = Path(__file__).resolve()


def stl_gap(fine: np.ndarray) -> float:
    """Largest difference between the fine values and the main STL surface, at the fine points between main points."""
    c = fine[::2, ::2]
    d = [np.abs(fine[::2, 1::2] - (c[:, :-1] + c[:, 1:]) / 2),  # halfway along x
         np.abs(fine[1::2, ::2] - (c[:-1, :] + c[1:, :]) / 2),  # halfway along y
         np.abs(fine[1::2, 1::2] - (c[:-1, :-1] + c[1:, 1:]) / 2)]  # halfway along the diagonal
    return float(max(x.max() for x in d))


def max_slope_deg(z: np.ndarray, L: float) -> float:
    n = z.shape[0] - 1
    gy, gx = np.gradient(z, L / n)
    return float(np.degrees(np.arctan(np.hypot(gx, gy).max())))


def max_face_slope_deg(z: np.ndarray, L: float) -> float:
    """Largest angle from the vertical of the normals of the top triangles of the STL surface of z."""
    tris = relief.solid(z - z.min() + 1.0, L)  # lifted above 0 for the solid; the angles do not depend on it
    n = (z.shape[0] - 1) ** 2 * 2
    nz = relief.normals(tris[:n])[:, 2]
    return float(np.degrees(np.arccos(np.clip(nz.min(), -1.0, 1.0))))


def main() -> None:
    if OUT.exists() or STL_DIR.exists():
        raise SystemExit(f"{OUT} or {STL_DIR} exists; refusing to overwrite")
    raw = COND.read_bytes()
    c = json.loads(raw)
    cm.same_as_head([COND, SCRIPT, ROOT / "src" / "relief.py", ROOT / "works" / "_common.py"])
    state = cm.git_state(["checks", "src", "works", "pyproject.toml", "uv.lock", ".python-version"])
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    cm.check_generator(c["generator"])
    d = relief.Dimensions(**c["dimensions"])
    nm, nf = c["grids"]["main_intervals"], c["grids"]["fine_intervals"]
    if nm != relief.grid_intervals(d) or nf != 2 * nm:
        raise SystemExit("the grids differ from those of the decisions")
    g1 = c["eps"]["E1"]["G_star"]
    if g1 != relief.G_STAR:
        raise SystemExit("G* of E1 differs from src/relief.py")
    g2 = 2 * math.pi * 7.0 * 4 / 1.875
    eps = {"E1": relief.eps_for(d, g1), "E2": relief.eps_for(d, g2)}
    seed = c["seed"]
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    pts = {}
    for p in c["points"]:
        Ff = lambda f: relief.closed_field(f, nf, seed)  # noqa: E731
        t1 = relief.tile(p["V"], p["A"], Ff, eps["E1"])
        t2 = relief.tile(p["V"], p["A"], Ff, eps["E2"])
        z1, z2 = relief.scaled(t1.H_eps, "common", d), relief.scaled(t2.H_eps, "common", d)
        zh = relief.scaled(t1.H, "common", d)
        zm = relief.scaled(t1.H_eps, "minmax", d)
        pts[p["name"]] = {"f": t1.f, "s": t1.s, "gamma": t1.gamma,
                          "R1_stl_gap_mm": {"H_eps_E1_common": stl_gap(z1), "H_eps_E1_minmax": stl_gap(zm),
                                            "H_common": stl_gap(zh)},
                          "R2": {"max_abs_E2_minus_E1_mm": float(np.abs(z2 - z1).max()),
                                 "max_slope_deg": {"E1": max_slope_deg(z1, d.L_mm), "E2": max_slope_deg(z2, d.L_mm),
                                                   "H": max_slope_deg(zh, d.L_mm)},
                                 "max_face_slope_deg": {"E1": max_face_slope_deg(z1, d.L_mm),
                                                        "E2": max_face_slope_deg(z2, d.L_mm),
                                                        "H": max_face_slope_deg(zh, d.L_mm)},
                                 "span_mm": {"E1": float(np.ptp(z1)), "E2": float(np.ptp(z2))}}}
    STL_DIR.mkdir(parents=True)
    r3 = {}
    hi = next(p for p in c["points"] if p["name"] == "high")
    for n in (nm, nf):
        t = relief.tile(hi["V"], hi["A"], lambda f: relief.closed_field(f, n, seed), eps["E1"])
        top = relief.top_surface(t.H_eps, "common", d)
        path = STL_DIR / f"high_common_{n}.stl"
        relief.write_stl(path, relief.solid(top, d.L_mm), f"supplement high common E1 {n} intervals")
        _, back = relief.read_stl(path)
        v = back.reshape(-1, 3)
        ec, vol = relief.edge_check(back), relief.signed_volume(back)
        if (ec["not_two"] or ec["same_direction"] or not vol > 0 or v[:, :2].min() != 0 or v[:, 2].min() != 0
                or abs(v[:, :2].max() - d.L_mm) > 1e-4 or v[:, 2].max() > d.base_mm + d.D_mm + 1e-4):
            raise SystemExit(f"the STL of {n} intervals fails its check: {ec}, volume {vol}")
        r3[str(n)] = {"file": path.name, "sha256": cm.sha256(path), "bytes": path.stat().st_size, "facets": len(back),
                      "edge_check": relief.edge_check(back), "volume_mm3": relief.signed_volume(back),
                      "min": v.min(axis=0).tolist(), "max": v.max(axis=0).tolist()}
    res = {"meta": {"conditions_sha256": hashlib.sha256(raw).hexdigest(), "script_sha256": cm.sha256(SCRIPT),
                    "relief_sha256": cm.sha256(ROOT / "src" / "relief.py"), **cm.versions("numpy"), **state,
                    "started": started, "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
           "eps": eps, "G_star": {"E1": g1, "E2": g2}, "points": pts, "R3": r3}
    with OUT.open("x") as fh:
        fh.write(json.dumps(res, indent=1, allow_nan=False) + "\n")
    print(json.dumps({k: res[k] for k in ("eps", "points")}, indent=1))


if __name__ == "__main__":
    main()
