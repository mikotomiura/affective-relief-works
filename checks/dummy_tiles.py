"""Series of dummy tiles for fixing the range of the number of sentences of work 1 (no text, no lexicon).

Usage:  uv run python checks/dummy_tiles.py            (refuses to overwrite)
Conditions: checks/conditions/dummy_tiles.json (committed before this script was run)
Output: results/dummy_tiles.json and results/dummy_tiles/n{n}_{export}.png
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import platform
import subprocess
from importlib import metadata
from pathlib import Path

import numpy as np
from PIL import Image

from atact.field import procedural_field_separable
from atact.mapping import F0, F1, params
from atact.transform import t_gamma

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
COND = ROOT / "checks" / "conditions" / "dummy_tiles.json"
OUT = ROOT / "results" / "dummy_tiles.json"
OUT_DIR = ROOT / "results" / "dummy_tiles"
OCTAVES = 4
MM_PER_INCH = 25.4


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_state() -> dict:
    def run(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()

    return {"git_commit": run("rev-parse", "HEAD"),
            "git_dirty": bool(run("status", "--porcelain", "--", "checks", "src", "pyproject.toml", "uv.lock"))}


def installed_commit() -> str:
    return json.loads(metadata.distribution("atact").read_text("direct_url.json"))["vcs_info"]["commit_id"]


def layout(n: int, fig: dict, pr: dict, L: float) -> dict:
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    tile_mm = (fig["width_mm"] - (cols - 1) * fig["gap_mm"]) / cols
    height_mm = rows * tile_mm + (rows - 1) * fig["gap_mm"]
    per_plate = math.floor((pr["bed_mm"] + pr["gap_mm"]) / (L + pr["gap_mm"])) ** 2
    return {"columns": cols, "rows": rows, "tile_mm": tile_mm, "figure_height_mm": height_mm,
            "tile_px": int(tile_mm / MM_PER_INCH * fig["dpi"]),
            "base_period_mm_at_f_max": tile_mm / (F0 + F1), "finest_octave_period_mm": tile_mm / ((F0 + F1) * 2 ** (OCTAVES - 1)),
            "tiles_per_plate": per_plate, "plates": math.ceil(n / per_plate),
            "board_mm": [cols * (L + pr["gap_mm"]) - pr["gap_mm"], rows * (L + pr["gap_mm"]) - pr["gap_mm"]]}


def grey(VA: np.ndarray, px: int, seed: int, d: dict) -> dict[str, list[np.ndarray]]:
    out = {"common": [], "minmax": []}
    k = d["D_mm"] / (2 * d["s_max"])
    for V, A in VA:
        f, s, g = (float(x) for x in params(V, A))
        H = s * t_gamma(procedural_field_separable(f, px, seed, octaves=OCTAVES, normalize="ampsum"), g)
        out["common"].append(k * H / d["D_mm"] + 0.5)
        out["minmax"].append((H - H.min()) / (H.max() - H.min()))
    return out


def figure(tiles: list[np.ndarray], lay: dict, fig: dict) -> Image.Image:
    dpmm = fig["dpi"] / MM_PER_INCH
    W = int(round(fig["width_mm"] * dpmm))
    Hpx = int(round(lay["figure_height_mm"] * dpmm))
    canvas = np.ones((Hpx, W))  # white background
    step = (lay["tile_mm"] + fig["gap_mm"]) * dpmm
    px = lay["tile_px"]
    for i, t in enumerate(tiles):
        r, c = divmod(i, lay["columns"])
        y, x = int(round(r * step)), int(round(c * step))
        canvas[y : y + px, x : x + px] = np.flipud(np.clip(t, 0.0, 1.0))  # origin lower, as in the text example
    return Image.fromarray(np.round(canvas * 255).astype(np.uint8), mode="L")


def main() -> None:
    if OUT.exists() or OUT_DIR.exists():
        raise SystemExit(f"{OUT} or {OUT_DIR} exists; refusing to overwrite")
    raw = COND.read_bytes()
    c = json.loads(raw)
    if installed_commit() != c["generator"]["commit"]:
        raise SystemExit("atact is not installed from the commit of the conditions")
    state = git_state()
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    co = c["coordinates"]
    VA_all = np.random.default_rng(co["seed"]).uniform(-1, 1, size=(co["n_max"], 2))
    d, fig, pr = c["dimensions"], c["figure"], c["print"]
    OUT_DIR.mkdir(parents=True)
    series = {}
    for n in c["series_sizes"]:
        lay = layout(n, fig, pr, d["L_mm"])
        g = grey(VA_all[:n], lay["tile_px"], c["surface"]["seed"], d)
        files = {}
        for e, tiles in g.items():
            p = OUT_DIR / f"n{n}_{e}.png"
            figure(tiles, lay, fig).save(p, dpi=(fig["dpi"], fig["dpi"]))
            files[e] = {"file": str(p.relative_to(ROOT)), "sha256": sha256(p)}
        lay["criteria"] = {"C1_tile_at_least_12_mm": lay["tile_mm"] >= 12.0,
                           "C2_base_period_at_least_1.5_mm": lay["base_period_mm_at_f_max"] >= 1.5,
                           "C3_fits_page_height": lay["figure_height_mm"] <= fig["max_height_mm"],
                           "C4": "by eye, recorded in the decisions (not computed)"}
        lay["common_grey_range"] = [float(min(t.min() for t in g["common"])), float(max(t.max() for t in g["common"]))]
        lay["files"] = files
        series[str(n)] = lay
    tiles = [{"V": float(V), "A": float(A), **dict(zip(("f", "s", "gamma"), (float(x) for x in params(V, A))))}
             for V, A in VA_all]
    res = {"meta": {"conditions_sha256": hashlib.sha256(raw).hexdigest(), "script_sha256": sha256(SCRIPT),
                    "atact_commit": installed_commit(), "numpy": np.__version__, "pillow": Image.__version__,
                    "python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}",
                    **state, "started": started,
                    "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
           "series": series, "tiles": tiles}
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    for n, s in series.items():
        print(n, {k: s[k] for k in ("columns", "rows", "tile_mm", "figure_height_mm", "base_period_mm_at_f_max", "plates", "board_mm")}, s["criteria"])


if __name__ == "__main__":
    main()
