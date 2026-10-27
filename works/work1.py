"""Work 1: one English story, one tile per sentence, under the common height scale and per-tile min-max.

Usage:  uv run python works/work1.py        (refuses to overwrite; stops if an input is still "FILL")
Conditions: works/conditions/work1.json (the template works/conditions/work1.template.json with the inputs of the
        chosen story filled in, committed before this script is run on it)
Input:  the Project Gutenberg file and the NRC VAD Lexicon v2.1 (external/, not in the repository)
Output: results/work1/work1.json and the figures results/work1/{export}.png (in the repository); the STL files
        output/work1/stl/s{i:03d}_{scale}.stl (outside the repository, their SHA-256 in work1.json)

Steps: src/pipeline.work1_text (extract_story -> split_sentences -> atact sentence_point -> (f, s, gamma)), then per
sentence the tile of src/relief.py: STL from H_eps on the print grid (points e_min / 2 apart) under both scales, and
the grey figures of the series (common and min-max of H, common of H_eps) on the grid of the printed figure.
An unmatched sentence (no term of the lexicon) is made from (0, 0), as the generator does, and flagged in the record.
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
sys.pycache_prefix = tempfile.mkdtemp(prefix="pycache-")  # never run a stale .pyc of src/ (ignored by git)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "works"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import _common as cm  # noqa: E402
import relief  # noqa: E402
from atact.field import procedural_field_separable  # noqa: E402
from atact.lexicon import Lexicon, load_lexicon  # noqa: E402
from atact.mapping import F0, F1, S0, S1  # noqa: E402
from english_text import extract_story, split_sentences  # noqa: E402
from pipeline import work1_text  # noqa: E402

COND = ROOT / "works" / "conditions" / "work1.json"
TEMPLATE = ROOT / "works" / "conditions" / "work1.template.json"
OUT_DIR = ROOT / "results" / "work1"
STL_DIR = ROOT / "output" / "work1" / "stl"
MM_PER_INCH = 25.4
FIGURES = (("common", "H", "common"), ("minmax", "H", "minmax"), ("common_eps", "H_eps", "common"))
CODE = ["works/work1.py", "works/_common.py", "src/relief.py", "src/pipeline.py", "src/english_text.py",
        "src/japanese_text.py", "src/window.py", "src/stretch.py"]


def dimensions(c: dict) -> relief.Dimensions:
    r = c["relief"]
    return relief.Dimensions(*(cm.need(r, k, float, "relief.") for k in ("L_mm", "D_mm", "s_max", "base_mm", "e_min_mm")))


def validate(c: dict) -> None:
    """Types and ranges of the conditions, and the values that follow from others (eps, the grid)."""
    t = c["text"]
    for k in ("file", "sha256", "encoding", "heading", "body_sha256"):
        cm.need(t, k, str, "text.")
    if t["encoding"] != "utf-8":
        raise SystemExit("the Gutenberg file is read as UTF-8")
    if t["next_heading"] is not None:
        cm.need(t, "next_heading", str, "text.")
    for k in ("occurrence", "heading_line", "sentences"):
        cm.need(t, k, int, "text.")
    lo, hi = cm.need(c["range"], "min_sentences", int), cm.need(c["range"], "max_sentences", int)
    if not lo <= t["sentences"] <= hi:
        raise SystemExit(f"{t['sentences']} sentences, outside the range {lo}-{hi} of work 1")
    r = c["relief"]
    d = dimensions(c)
    if d.s_max < S0 + S1:
        raise SystemExit("s_max is below the largest s of the rule; the common scale would leave the span D")
    for k in ("G_star", "eps"):
        cm.need(r, k, float, "relief.")
    if r["G_star"] != relief.G_STAR or not math.isclose(r["eps"], relief.eps_for(d, r["G_star"]), rel_tol=1e-12):
        raise SystemExit("eps is not e_min G* / (2 L) with the G* of the paper on the rule")
    if cm.need(r, "grid_intervals", int, "relief.") != relief.grid_intervals(d):
        raise SystemExit("the grid is not L / (e_min / 2) intervals")
    if (cm.need(r, "octaves", int, "relief.") != relief.OCTAVES or r.get("normalize") != relief.NORMALIZE
            or cm.need(r, "seed", int, "relief.") < 0):
        raise SystemExit("the field settings differ from those of src/relief.py")
    f = c["figure"]
    for k in ("width_mm", "gap_mm"):
        cm.need(f, k, float, "figure.")
    cm.need(f, "dpi", int, "figure.")


def layout(n: int, fig: dict) -> dict:
    """Columns ceil(sqrt n), as for the dummy tiles (checks/dummy_tiles.py)."""
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    tile_mm = (fig["width_mm"] - (cols - 1) * fig["gap_mm"]) / cols
    return {"columns": cols, "rows": rows, "tile_mm": tile_mm,
            "figure_height_mm": rows * tile_mm + (rows - 1) * fig["gap_mm"],
            "tile_px": int(tile_mm / MM_PER_INCH * fig["dpi"]), "base_period_mm_at_f_max": tile_mm / (F0 + F1)}


def figure(tiles: list[np.ndarray], lay: dict, fig: dict) -> Image.Image:
    dpmm = fig["dpi"] / MM_PER_INCH
    canvas = np.ones((int(round(lay["figure_height_mm"] * dpmm)), int(round(fig["width_mm"] * dpmm))))
    step = (lay["tile_mm"] + fig["gap_mm"]) * dpmm
    px = lay["tile_px"]
    for i, t in enumerate(tiles):
        r, c = divmod(i, lay["columns"])
        y, x = int(round(r * step)), int(round(c * step))
        canvas[y : y + px, x : x + px] = np.flipud(np.clip(t, 0.0, 1.0))  # origin lower left
    return Image.fromarray(np.round(canvas * 255).astype(np.uint8), mode="L")


def run(c: dict, root: Path, lex: Lexicon, out_dir: Path, stl_dir: Path) -> dict:
    """The work from the conditions; the inputs are read from root. Writes the figures and the STL, returns the record."""
    validate(c)
    t, r, fig = c["text"], c["relief"], c["figure"]
    raw = cm.check_file(root / t["file"], t["sha256"], "the text file")
    text = raw.decode("utf-8")
    # the extraction is checked before the lexicon is used on the text
    paras, rep = extract_story(text, t["heading"], t["next_heading"], t["occurrence"])
    if rep.start_line != t["heading_line"] or rep.body_sha256 != t["body_sha256"]:
        raise SystemExit("the extracted story differs from the one of the conditions (line or SHA-256 of the body)")
    n_sent = len(split_sentences(paras)[0])
    if n_sent != t["sentences"]:
        raise SystemExit(f"{n_sent} sentences, not {t['sentences']}")
    w = work1_text(text, t["heading"], t["next_heading"], t["occurrence"], lex)
    d = dimensions(c)
    eps, n, seed = r["eps"], r["grid_intervals"], r["seed"]
    lay = layout(len(w.points), fig)
    out_dir.mkdir(parents=True)
    stl_dir.mkdir(parents=True)
    grey = {name: [] for name, _, _ in FIGURES}
    for i, p in enumerate(w.points):
        tp = relief.tile(p["V"], p["A"], lambda f: relief.closed_field(f, n, seed), eps)
        p["stl"] = {}
        for scale in relief.SCALES:
            top = relief.top_surface(tp.H_eps, scale, d)
            path = stl_dir / f"s{i:03d}_{scale}.stl"
            if not (d.base_mm - 1e-9 <= top.min() and top.max() <= d.base_mm + d.D_mm + 1e-9):
                raise SystemExit(f"sentence {i}: the top leaves [b, b + D]")
            tris = relief.solid(top, d.L_mm)
            relief.write_stl(path, tris, f"work 1 sentence {i} {scale} H_eps L {d.L_mm} mm")
            _, back = relief.read_stl(path)  # the file as written (float32): closed, outward, of the right size
            ec, vol = relief.edge_check(back), relief.signed_volume(back)
            v = back.reshape(-1, 3)
            if (ec["not_two"] or ec["same_direction"] or not vol > 0 or len(back) != len(tris)
                    or v[:, :2].min() != 0 or abs(v[:, :2].max() - d.L_mm) > 1e-4):
                raise SystemExit(f"sentence {i}: the STL {path.name} fails its check {ec}, volume {vol}")
            p["stl"][scale] = {"file": path.name, "sha256": cm.sha256(path), "facets": len(tris), "volume_mm3": vol,
                               "z_min_mm": float(top.min()), "z_max_mm": float(top.max())}
        tf = relief.tile(p["V"], p["A"], lambda f: procedural_field_separable(
            f, lay["tile_px"], seed, octaves=relief.OCTAVES, normalize=relief.NORMALIZE), eps)
        for name, field, scale in FIGURES:
            grey[name].append(relief.grey(getattr(tf, field), scale, d))
    figs = {}
    for name, tiles in grey.items():
        path = out_dir / f"{name}.png"
        figure(tiles, lay, fig).save(path, dpi=(fig["dpi"], fig["dpi"]))
        figs[name] = {"file": path.name, "sha256": cm.sha256(path)}
    rep = w.report
    return {"story": {"start_line": rep.start_line, "end_line": rep.end_line, "paragraphs": rep.paragraphs,
                      "dropped_brackets": rep.dropped_brackets, "body_sha256": rep.body_sha256,
                      "dropped_pieces": w.dropped},
            "eps": eps, "grid_intervals": n, "layout": lay, "figures": figs,
            "unmatched": sum(p["unmatched"] for p in w.points),
            "matched_share": sum(not p["unmatched"] for p in w.points) / len(w.points), "sentences": w.points}


def main() -> None:
    out_json = OUT_DIR / "work1.json"
    if OUT_DIR.exists() or STL_DIR.exists():
        raise SystemExit(f"{OUT_DIR} or {STL_DIR} exists; refusing to overwrite")
    raw, c = cm.load_conditions(COND)
    validate(c)
    cm.same_as_head([COND, TEMPLATE, *(ROOT / p for p in CODE)])
    cm.check_template(TEMPLATE, c)
    state = cm.git_state(["works", "src", "pyproject.toml", "uv.lock", ".python-version"])
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    cm.check_generator(c["generator"])
    lex = load_lexicon(ROOT / c["lexicon"]["file"])
    if lex.sha256 != c["lexicon"]["sha256"]:
        raise SystemExit("the lexicon does not match the SHA-256 of the conditions")
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    res = run(c, ROOT, lex, OUT_DIR, STL_DIR)
    res = {"meta": {"conditions_sha256": hashlib.sha256(raw).hexdigest(), "lexicon_sha256": lex.sha256,
                    "atact_library_sha256": cm.library_sha256(), "uv_lock_sha256": cm.sha256(ROOT / "uv.lock"),
                    **cm.versions("numpy", "pillow"), **state, "started": started,
                    "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}, **res}
    with out_json.open("x") as fh:
        fh.write(json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({k: res[k] for k in ("eps", "grid_intervals", "unmatched", "matched_share", "figures")}, indent=1))


if __name__ == "__main__":
    main()
