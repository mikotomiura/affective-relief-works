"""Strip sketch of work 1 in three versions, for choosing the direction of the work (a check, not a part of it).

Usage:  uv run python checks/strip_sketch.py            (refuses to overwrite)
Conditions: checks/conditions/strip_sketch.json (committed before this script was run)
Input:  results/work1/work1.json (V and A of the sentences) and the Project Gutenberg file of work 1 (external/)
Output: results/strip_sketch.json and results/strip_sketch/{v1,v2,v3,v2_key}.png

The sentences are laid out as one continuous strip (no gap inside a row). v1 keeps one seed for every tile, as
work 1; v2 and v3 change the seed with the author's paragraph (content), while (f, s, gamma) still come from (V, A).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atact.field import procedural_field_separable  # noqa: E402
from atact.mapping import params  # noqa: E402
from atact.transform import t_gamma  # noqa: E402
from english_text import _has_word_char, extract_story, split_paragraph  # noqa: E402

SCRIPT = Path(__file__).resolve()
COND = ROOT / "checks" / "conditions" / "strip_sketch.json"
OUT = ROOT / "results" / "strip_sketch.json"
OUT_DIR = ROOT / "results" / "strip_sketch"
OCTAVES = 4
BASE_SEED = 42
MM_PER_INCH = 25.4
KEY_MARK_MM = 0.5


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_state() -> dict:
    def run(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()

    return {"git_commit": run("rev-parse", "HEAD"),
            "git_dirty": bool(run("status", "--porcelain", "--", "checks", "src", "works", "results/work1",
                                  "pyproject.toml", "uv.lock"))}


def installed_commit() -> str:
    return json.loads(metadata.distribution("atact").read_text("direct_url.json"))["vcs_info"]["commit_id"]


def paragraph_of_sentences(text: str, inp: dict) -> tuple[list[int], list[str]]:
    """Paragraph index of every sentence, in order, and the sentences (the rule of split_sentences)."""
    paras, _ = extract_story(text, inp["heading"], inp["next_heading"], inp["occurrence"])
    idx, sents = [], []
    for k, p in enumerate(paras):
        for s in split_paragraph(p):
            if _has_word_char(s):
                idx.append(k)
                sents.append(s)
    return idx, sents


def grey(V: float, A: float, px: int, seed: int, scale: str, d: dict) -> np.ndarray:
    f, s, g = (float(x) for x in params(V, A))
    H = s * t_gamma(procedural_field_separable(f, px, seed, octaves=OCTAVES, normalize="ampsum"), g)
    if scale == "common":
        return d["D_mm"] * H / (2 * d["s_max"]) / d["D_mm"] + 0.5
    if scale == "minmax":
        span = H.max() - H.min()
        if not span > 0:
            raise SystemExit("a flat tile has no min-max scale")
        return d["D_mm"] * (H - H.min()) / span / d["D_mm"]  # the order of operations of work 1
    raise SystemExit(f"unknown scale {scale!r}")


def layout(st: dict, n: int) -> dict:
    """Whole pixels: tiles of tile_px side by side with no seam, rows row_gap_px apart (the sizes as printed)."""
    if st["per_row"] * st["rows"] != n:
        raise SystemExit(f"{st['per_row']} x {st['rows']} is not {n} tiles")
    dpmm = st["dpi"] / MM_PER_INCH
    px = int(st["width_mm"] / st["per_row"] * dpmm)
    gap = int(round(st["row_gap_mm"] * dpmm))
    return {"tile_px": px, "row_gap_px": gap, "width_px": st["per_row"] * px,
            "height_px": st["rows"] * px + (st["rows"] - 1) * gap,
            "tile_mm": px / dpmm, "row_gap_mm": gap / dpmm, "width_mm": st["per_row"] * px / dpmm}


def figure(tiles: list[np.ndarray], st: dict, lay: dict, starts: set[int] | None = None) -> Image.Image:
    """The strip; with starts, a black mark on the left edge of every tile that opens a paragraph (not the first)."""
    px, gap = lay["tile_px"], lay["row_gap_px"]
    canvas = np.ones((lay["height_px"], lay["width_px"]))
    mark = max(1, int(round(KEY_MARK_MM * st["dpi"] / MM_PER_INCH)))
    for i, t in enumerate(tiles):
        r, c = divmod(i, st["per_row"])
        y, x = r * (px + gap), c * px
        canvas[y : y + px, x : x + px] = np.flipud(np.clip(t, 0.0, 1.0))  # origin lower left, as work 1
        if starts is not None and i in starts:
            canvas[y : y + px, x : x + mark] = 0.0
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
    inp, d, st = c["input"], c["dimensions"], c["strip"]
    pts_path, txt_path = ROOT / inp["points"], ROOT / inp["text"]
    if sha256(pts_path) != inp["points_sha256"] or sha256(txt_path) != inp["text_sha256"]:
        raise SystemExit("an input differs from its SHA-256 in the conditions")
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    pts = json.loads(pts_path.read_text())["sentences"]
    para, sents = paragraph_of_sentences(txt_path.read_text(encoding="utf-8"), inp)
    if sents != [p["text"] for p in pts]:
        raise SystemExit("the sentences of the paragraphs differ from the points of work 1")
    for p in pts:
        if not np.allclose([float(x) for x in params(p["V"], p["A"])], [p["f"], p["s"], p["gamma"]], rtol=0, atol=1e-12):
            raise SystemExit("(f, s, gamma) of work 1 differ from params(V, A)")
    if max(para) + 1 != c["versions"]["paragraphs"]:
        raise SystemExit(f"{max(para) + 1} paragraphs, not {c['versions']['paragraphs']}")
    lay = layout(st, len(pts))
    px = lay["tile_px"]
    seeds = {"v1": [BASE_SEED] * len(pts), "v2": [BASE_SEED + k for k in para]}
    seeds["v3"] = seeds["v2"]
    scales = {"v1": "common", "v2": "common", "v3": "minmax"}
    tiles = {v: [grey(p["V"], p["A"], px, seeds[v][i], scales[v], d) for i, p in enumerate(pts)] for v in scales}
    starts = {i for i in range(1, len(para)) if para[i] != para[i - 1]}
    OUT_DIR.mkdir(parents=True)
    files = {}
    for name, img in [*((v, figure(t, st, lay)) for v, t in tiles.items()), ("v2_key", figure(tiles["v2"], st, lay, starts))]:
        p = OUT_DIR / f"{name}.png"
        img.save(p, dpi=(st["dpi"], st["dpi"]))
        files[name] = {"file": str(p.relative_to(ROOT)), "sha256": sha256(p)}
    sizes = np.bincount(para).tolist()
    res = {"meta": {"conditions_sha256": hashlib.sha256(raw).hexdigest(), "script_sha256": sha256(SCRIPT),
                    "atact_commit": installed_commit(), "numpy": np.__version__, "pillow": Image.__version__,
                    "python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}",
                    **state, "started": started,
                    "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
           "layout": lay, "paragraphs": len(sizes), "sentences_per_paragraph": sizes,
           "paragraph_starts": sorted(starts), "files": files,
           "sentences": [{"i": i, "paragraph": para[i], "seed_v2": seeds["v2"][i], "V": p["V"], "A": p["A"],
                          "f": p["f"], "s": p["s"], "gamma": p["gamma"]} for i, p in enumerate(pts)]}
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    print(lay, "paragraphs", len(sizes), sizes, {k: v["file"] for k, v in files.items()})


if __name__ == "__main__":
    main()
