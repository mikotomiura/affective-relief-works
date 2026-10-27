"""Work 1, the drawing: the first twelve tiles of work 1 drawn four ways (height as grey, lit from the upper left, lit
from the lower right, lit and stretched per tile), from the recorded values.

Usage:  uv run python works/work1_drawing.py        (refuses to overwrite)
Conditions: works/conditions/work1_drawing.json, and the relief, seed and figure of works/conditions/work1.json
        (committed before this script was run)
Input:  results/work1/work1.json and results/work1/common_eps.png (no text, no lexicon)
Output: results/work1_drawing/drawing.json and drawing.png
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.pycache_prefix = tempfile.mkdtemp(prefix="pycache-")  # never run a stale .pyc of src/ (ignored by git)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "works"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from scipy import stats  # noqa: E402

import _common as cm  # noqa: E402
import relief  # noqa: E402
from atact.field import procedural_field_separable  # noqa: E402
from atact.mapping import params  # noqa: E402

COND = ROOT / "works" / "conditions" / "work1_drawing.json"
WORK1_COND = ROOT / "works" / "conditions" / "work1.json"
OUT_DIR = ROOT / "results" / "work1_drawing"
CODE = ["works/work1_drawing.py", "works/work1.py", "works/_common.py", "src/relief.py"]
MM_PER_INCH = 25.4


def lambert(z: np.ndarray, side_mm: float, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """max(0, n . l) of the surface z (mm, rows y, columns x, spacing side / n), light at the azimuth (counterclockwise
    from +x, with +y up) and elevation given."""
    h = side_mm / z.shape[0]
    gy, gx = np.gradient(z, h)
    n = np.stack([-gx, -gy, np.ones_like(z)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    a, e = math.radians(azimuth_deg), math.radians(elevation_deg)
    light = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    return np.clip(n @ light, 0.0, 1.0)


def block_mean(x: np.ndarray, k: int) -> np.ndarray:
    n = x.shape[0] // k
    if x.shape != (n * k, n * k):
        raise ValueError(f"shape {x.shape} is not a square of side divisible by {k}")
    return x.reshape(n, k, n, k).mean(axis=(1, 3))


def stretch(g: np.ndarray) -> np.ndarray:
    span = g.max() - g.min()
    if not span > 0:
        raise ValueError("a flat tile cannot be stretched")
    return (g - g.min()) / span


def tile_place(i: int, lay: dict, fig: dict) -> tuple[int, int]:
    """Top left (row, column) of tile i in the figures of work 1 (works/work1.py figure)."""
    step = (lay["tile_mm"] + fig["gap_mm"]) * fig["dpi"] / MM_PER_INCH
    r, c = divmod(i, lay["columns"])
    return int(round(r * step)), int(round(c * step))


def u8(g: np.ndarray) -> np.ndarray:
    """Grey 0..1 (rows y up) to an image (rows from the top)."""
    return np.round(np.flipud(np.clip(g, 0.0, 1.0)) * 255).astype(np.uint8)


def figure(rows: list[tuple[str, list[np.ndarray]]], f: dict, px: int) -> Image.Image:
    gap, rl, rg = f["gap_px"], f["row_label_px"], f["row_gap_px"]
    n = max(len(t) for _, t in rows)
    W = n * px + (n - 1) * gap
    H = len(rows) * (rl + px) + (len(rows) - 1) * rg
    img = Image.new("L", (W, H), 255)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(f["font_file"], 22)
    for r, (label, tiles) in enumerate(rows):
        y = r * (rl + px + rg)
        draw.text((0, y + 4), label, fill=0, font=font)
        for k, g in enumerate(tiles):
            img.paste(Image.fromarray(u8(g), mode="L"), (k * (px + gap), y + rl))
    return img


def main() -> None:
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} exists; refusing to overwrite")
    raw = COND.read_bytes()
    c = json.loads(raw)
    inp = c["inputs"]
    w1_path, ce_path = ROOT / inp["work1"], ROOT / inp["common_eps"]
    for p, h in ((w1_path, inp["work1_sha256"]), (WORK1_COND, inp["work1_conditions_sha256"])):
        if cm.sha256(p) != h:
            raise SystemExit(f"{p.relative_to(ROOT)} differs from the one named in the conditions")
    cm.same_as_head([COND, WORK1_COND, w1_path, ce_path, *(ROOT / p for p in CODE)])
    state = cm.git_state(["works", "src", "results/work1", "pyproject.toml", "uv.lock"])
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    w1c = json.loads(WORK1_COND.read_bytes())
    w1 = json.loads(w1_path.read_bytes())
    if w1["meta"]["conditions_sha256"] != inp["work1_conditions_sha256"]:
        raise SystemExit("work1.json was not computed on the conditions named")
    if w1["figures"]["common_eps"]["sha256"] != cm.sha256(ce_path):
        raise SystemExit("common_eps.png differs from the record")
    cm.check_generator(w1c["generator"])
    r = w1c["relief"]
    d = relief.Dimensions(r["L_mm"], r["D_mm"], r["s_max"], r["base_mm"], r["e_min_mm"])
    eps = relief.eps_for(d)
    if eps != w1["eps"]:
        raise SystemExit("eps differs from the record")
    lay, fig1 = w1["layout"], w1c["figure"]
    px, k, seed = lay["tile_px"], c["drawings"]["supersample"], r["seed"]
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    ce = np.asarray(Image.open(ce_path))
    dr = c["drawings"]
    el, az_b, az_c = dr["elevation_deg"], dr["azimuth_b_deg"], dr["azimuth_c_deg"]

    def heights(p: dict, n: int) -> np.ndarray:
        t = relief.tile(p["V"], p["A"], lambda f: procedural_field_separable(
            f, n, seed, octaves=relief.OCTAVES, normalize=relief.NORMALIZE), eps)
        return t.H_eps

    out = {name: [] for name in ("a", "b", "c", "d")}
    per_tile = []
    for i in range(c["tiles"]["n"]):
        p = w1["sentences"][i]
        if [float(x) for x in params(p["V"], p["A"])] != [p["f"], p["s"], p["gamma"]]:
            raise SystemExit(f"sentence {i}: params differ from the record")
        a = relief.grey(heights(p, px), "common", d)
        y, x = tile_place(i, lay, fig1)
        if not np.array_equal(u8(a), ce[y : y + px, x : x + px]):  # G1
            raise SystemExit(f"sentence {i}: drawing a differs from its place in common_eps.png")
        z = relief.scaled(heights(p, px * k), "common", d)
        b = block_mean(lambert(z, d.L_mm, az_b, el), k)
        cc = block_mean(lambert(z, d.L_mm, az_c, el), k)
        dd = stretch(b)
        for name, g in zip("abcd", (a, b, cc, dd)):
            out[name].append(g)
        per_tile.append({"sentence": i, "s": p["s"], **{name: {"mean": float(g.mean()), "sd": float(g.std())}
                                                          for name, g in zip("abcd", (a, b, cc, dd))}})
    rho = {name: float(stats.spearmanr([t["s"] for t in per_tile], [t[name]["sd"] for t in per_tile]).statistic)
           for name in "abcd"}
    tmp = Path(tempfile.mkdtemp(prefix=".work1_drawing-", dir=OUT_DIR.parent))
    try:
        f = c["figure"]
        figure(list(zip(f["rows"], (out[n] for n in "abcd"))), f, px).save(tmp / f["file"])
        res = {"meta": {"conditions_sha256": hashlib.sha256(raw).hexdigest(), "script_sha256": cm.sha256(Path(__file__)),
                        "font_sha256": cm.sha256(Path(f["font_file"])), **cm.versions("numpy", "scipy", "pillow"),
                        **state, "started": started,
                        "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
               "eps": eps, "a_equals_common_eps": True, "per_tile": per_tile, "spearman_s_sd": rho, "G4_rho_d_lt_b": rho["d"] < rho["b"],
               "figures": {f["file"]: cm.sha256(tmp / f["file"])}}
        text = json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
        with (tmp / "drawing.json").open("x") as fh:
            fh.write(text)
    except BaseException:
        shutil.rmtree(tmp)
        raise
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} appeared during the run; the output is left in {tmp}")
    tmp.rename(OUT_DIR)
    print(json.dumps(rho, indent=1))


if __name__ == "__main__":
    main()
