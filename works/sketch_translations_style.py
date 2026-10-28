"""Sketch of a series of translators, the strip of the share of function words: under a row of relief tiles of each
translation (the tiles of version 1, each run its mean V and A, at N runs), a strip of N cells whose grey is the share
of the run's tokens that are in a fixed external list of function words. A permutation of the translation labels in
blocks of runs gives a reference for whether the mean shares of the three translations can be told from label noise.

Usage:  uv run python works/sketch_translations_style.py        (refuses to overwrite)
Conditions: works/conditions/sketch_translations_style.json, with the texts, extraction, tokenize, the cut into runs,
        lexicon, generator and relief of works/conditions/sketch_translations.json (both committed before this script
        is run)
Input:  the three texts, the NRC VAD Lexicon v2.1 and the list of function words (external/, not in the repository)
Output: results/sketch_translations_style/sketch_translations_style.json and the figures (written before the record;
        if a run fails between them, the directory is removed by hand before the run is repeated)

The share of a run is (tokens of the run in the list) / (tokens of the run), the tokens of atact.lexicon.tokenize (a
contraction such as "don't" is one token, and is counted only if the list holds it as one token). The groups (parts of
speech) of the list are counted the same way, each word in one group.
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
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import _common as cm  # noqa: E402
import relief  # noqa: E402
import sketch_translations as st  # noqa: E402
from atact.field import procedural_field_separable  # noqa: E402
from atact.lexicon import Lexicon, load_lexicon, tokenize  # noqa: E402

COND = ROOT / "works" / "conditions" / "sketch_translations_style.json"
BASE_COND = ROOT / "works" / "conditions" / "sketch_translations.json"
OUT_DIR = ROOT / "results" / "sketch_translations_style"
CODE = ["works/sketch_translations_style.py", "works/sketch_translations.py", "works/_common.py", "src/relief.py",
        "works/function_word_groups.py"]
VARIANTS = ("strip", "framed", "none")


# ---------------------------------------------------------------- the list

def read_list(raw: bytes) -> list[str]:
    """One word a line; empty lines and lines starting with '#' skipped; lower case, stripped."""
    words = [s.strip().lower() for s in raw.decode("utf-8").splitlines()]
    return [w for w in words if w and not w.startswith("#")]


def check_groups(words: list[str], groups: dict[str, list[str]]) -> dict[str, str]:
    """word -> group; every word of the list in exactly one group, no other word."""
    of = {}
    for g, ws in groups.items():
        for w in ws:
            if w in of:
                raise SystemExit(f"{w!r} is in two groups")
            of[w] = g
    if set(of) != set(words):
        raise SystemExit(f"the groups do not cover the list exactly: missing {sorted(set(words) - set(of))}, "
                         f"extra {sorted(set(of) - set(words))}")
    return of


# ---------------------------------------------------------------- the shares

def shares(tokens: list[str], n: int, words: set[str], group_of: dict[str, str], group_names: list[str]) -> dict:
    """Per run: tokens, function tokens, share, and the share of each group."""
    out = {"tokens": [], "function": [], "share": [], "groups": {g: [] for g in group_names}}
    for run in st.runs(tokens, n):
        f = [t for t in run if t in words]
        out["tokens"].append(len(run))
        out["function"].append(len(f))
        out["share"].append(len(f) / len(run))
        for g in group_names:
            out["groups"][g].append(sum(1 for t in f if group_of[t] == g) / len(run))
    return out


def statistic(x: np.ndarray) -> tuple[float, np.ndarray]:
    """x: (3, n) shares per run. Mean share per translation minus the mean of the three; returns the largest absolute
    departure and the departures."""
    m = x.mean(axis=1)
    d = m - m.mean()
    return float(np.abs(d).max()), d


def permute(x: np.ndarray, block: int, n_perm: int, seed: int) -> np.ndarray:
    """In each block of `block` consecutive runs the three labels are permuted (one permutation for every run of the
    block, independently between blocks); returns the statistic of every permutation."""
    n = x.shape[1]
    if n % block:
        raise SystemExit(f"{n} runs do not split into blocks of {block}")
    rng = np.random.default_rng(seed)
    out = np.empty(n_perm)
    cols = np.arange(n)
    for p in range(n_perm):
        perm = np.repeat(np.array([rng.permutation(3) for _ in range(n // block)]), block, axis=0)  # (n, 3)
        out[p] = statistic(x[perm.T, cols[None, :]])[0]
    return out


# ---------------------------------------------------------------- the figure

def strip_grey(v: np.ndarray, lo: float, hi: float) -> tuple[np.ndarray, int]:
    """Grey of a share: lo -> white (1), hi -> black (0), linear, clipped; returns the greys and the number clipped."""
    if not hi > lo:
        raise SystemExit(f"the scale has no width ({lo}, {hi})")
    g = 1.0 - (v - lo) / (hi - lo)
    return np.clip(g, 0.0, 1.0), int(np.sum((g < 0) | (g > 1)))


def figure(rows: list[tuple[list[np.ndarray], np.ndarray]], f: dict, n: int, variant: str) -> np.ndarray:
    """One row per translation: n relief tiles side by side (origin lower left), under them the strip of n cells,
    unless variant is 'none'. 'framed' draws a black line of frame_px around the strip, outside its cells; every
    variant has a white margin of frame_px on the left and right so that the three have the same width."""
    W, fp = f["row_width_px"], f["frame_px"]
    gap = f["tile_gap_px"]
    px = (W - (n - 1) * gap) // n
    sh = max(1, round(px * f["strip_height_ratio"]))
    row_h = px + (0 if variant == "none" else f["strip_gap_px"] + sh + fp)  # room for the lower frame
    H = len(rows) * row_h + (len(rows) - 1) * f["row_gap_px"]
    a = np.ones((H, W + 2 * fp))
    for i, (tiles, greys) in enumerate(rows):
        y = i * (row_h + f["row_gap_px"])
        for k, t in enumerate(tiles):
            x = fp + k * (px + gap)
            a[y : y + px, x : x + px] = np.flipud(np.clip(t, 0.0, 1.0))
        if variant == "none":
            continue
        ys = y + px + f["strip_gap_px"]
        for k, g in enumerate(greys):
            x = fp + k * (px + gap)
            a[ys : ys + sh, x : x + px] = g
        if variant == "framed":
            x1 = fp + (n - 1) * (px + gap) + px  # right edge of the last cell
            a[ys - fp : ys, 0 : x1 + fp] = 0.0
            a[ys + sh : ys + sh + fp, 0 : x1 + fp] = 0.0
            a[ys - fp : ys + sh + fp, 0 : fp] = 0.0
            a[ys - fp : ys + sh + fp, x1 : x1 + fp] = 0.0
    return a


def legend(lo: float, hi: float, f: dict, font_path: str, title: str) -> Image.Image:
    """A bar from lo (white) to hi (black) with the share written at both ends and at the middle."""
    w, h, pad = f["legend_width_px"], f["legend_bar_px"], 10
    font = ImageFont.truetype(font_path, f["font_size"])
    img = Image.new("L", (w + 2 * pad, h + 3 * f["font_size"] + 3 * pad), 255)
    bar = np.tile(np.linspace(1.0, 0.0, w), (h, 1))
    img.paste(Image.fromarray(np.round(bar * 255).astype(np.uint8), mode="L"), (pad, pad + f["font_size"] + pad))
    d = ImageDraw.Draw(img)
    d.text((pad, pad), title, fill=0, font=font)
    y = 2 * pad + f["font_size"] + h + 4
    for frac in (0.0, 0.5, 1.0):
        s = f"{lo + frac * (hi - lo):.3f}"
        x = pad + int(frac * (w - 1)) - (0 if frac == 0 else d.textlength(s, font=font) // (1 if frac == 1 else 2))
        d.text((x, y), s, fill=0, font=font)
    return img


def save(a: np.ndarray, path: Path, dpi: int) -> dict:
    Image.fromarray(np.round(a * 255).astype(np.uint8), mode="L").save(path, dpi=(dpi, dpi))
    return {"file": path.name, "sha256": cm.sha256(path)}


# ---------------------------------------------------------------- run

def validate(c: dict) -> None:
    fw = cm.need(c, "function_words", dict)
    for k in ("file", "sha256"):
        cm.need(fw, k, str, "function_words.")
    if cm.need(fw, "n_words", int, "function_words.") < 1:
        raise SystemExit("function_words.n_words is positive")
    groups = cm.need(c, "groups", dict)
    if not groups or not all(isinstance(v, list) and all(isinstance(w, str) for w in v) for v in groups.values()):
        raise SystemExit("groups maps a name to a list of words")
    cm.need(c, "groups_source", str)
    cm.need(c, "groups_script_sha256", str)
    n_main = cm.need(c, "runs_main", int)
    sens = cm.need(c, "runs_sensitivity", list)
    if not all(isinstance(x, int) and not isinstance(x, bool) and x >= 2 for x in sens) or n_main < 2 or n_main in sens:
        raise SystemExit("runs_main and runs_sensitivity are distinct integers >= 2")
    sc = cm.need(c, "scale", dict)
    if not 0 < cm.need(sc, "fixed_width", float, "scale.") < 1:
        raise SystemExit("scale.fixed_width is in (0, 1)")
    pl, ph = cm.need(sc, "percentile_low", float, "scale."), cm.need(sc, "percentile_high", float, "scale.")
    if not 0 <= pl < ph <= 100:
        raise SystemExit("0 <= percentile_low < percentile_high <= 100")
    p = cm.need(c, "permutation", dict)
    if cm.need(p, "n_permutations", int, "permutation.") < 100:
        raise SystemExit("at least 100 permutations")
    cm.need(p, "seed", int, "permutation.")
    blocks = cm.need(p, "blocks", list, "permutation.")
    if len(set(sens)) != len(sens):
        raise SystemExit("runs_sensitivity has a repeated value")
    if any(b.get("role") not in ("reading", "sensitivity") for b in blocks) or \
            sum(b.get("role") == "reading" for b in blocks) != 1:
        raise SystemExit("each block entry is a reading or a sensitivity, with exactly one reading")
    if not blocks or blocks[0].get("runs") != n_main or blocks[0].get("role") != "reading":
        raise SystemExit("the first entry of permutation.blocks is the reading, at runs_main")
    for b in blocks:
        n, bl = cm.need(b, "runs", int, "permutation.blocks."), cm.need(b, "block_runs", int, "permutation.blocks.")
        if n not in (n_main, *sens) or bl < 1 or n % bl:
            raise SystemExit(f"permutation block {b}: runs not drawn or not a multiple of block_runs")
        if math.factorial(3) ** (n // bl) < 20 * p["n_permutations"] and b.get("role") == "reading":
            raise SystemExit("the reading block leaves too few distinct permutations for its 95th percentile")
    f = cm.need(c, "figure", dict)
    for k in ("row_width_px", "tile_gap_px", "row_gap_px", "strip_gap_px", "frame_px", "dpi", "legend_width_px",
              "legend_bar_px", "font_size"):
        if cm.need(f, k, int, "figure.") < 1:
            raise SystemExit(f"figure.{k} is positive")
    if not 0 < cm.need(f, "strip_height_ratio", float, "figure.") <= 1:
        raise SystemExit("figure.strip_height_ratio is in (0, 1]")
    cm.need(f, "font_file", str, "figure.")
    if f["frame_px"] > f["strip_gap_px"]:
        raise SystemExit("the frame must fit in the gap above the strip")


def run(c: dict, base: dict, base_sha256: str, root: Path, lex: Lexicon, out_dir: Path) -> dict:
    st.validate(base)
    validate(c)
    if c["base_conditions_sha256"] != base_sha256:
        raise SystemExit("the conditions of the first sketch differ from the ones named here")
    fw = c["function_words"]
    words = read_list(cm.check_file(root / fw["file"], fw["sha256"], "the list of function words"))
    if len(words) != fw["n_words"] or len(set(words)) != len(words):
        raise SystemExit(f"the list holds {len(words)} words ({len(set(words))} distinct), not {fw['n_words']}")
    group_names = list(c["groups"])
    group_of = check_groups(words, c["groups"])
    cm.check_file(root / "works" / "function_word_groups.py", c["groups_script_sha256"], "the script of the groups")
    wset = set(words)
    r, f = base["relief"], c["figure"]
    d = st.dimensions(base)
    names = [t["name"] for t in base["translations"]]
    if len(names) != 3:
        raise SystemExit("the permutation is written for three translations")
    toks = {}
    for t in base["translations"]:  # every extraction is checked before anything is counted (as in version 1)
        text = cm.check_file(root / t["file"], t["sha256"], f"the text of {t['name']}").decode("utf-8")
        kept, dropped, line = st.extract_body(text, t["heading"])
        tokens = tokenize(" ".join(kept))
        last = hashlib.sha256(kept[-1].encode("utf-8")).hexdigest()
        if (line, st.body_sha256(kept), len(tokens), dropped, last) != (
                t["heading_line"], t["body_sha256"], t["tokens"], t["dropped_captions"], t["last_line_sha256"]):
            raise SystemExit(f"{t['name']}: the extracted body differs from the conditions")
        toks[t["name"]] = tokens
    ns = [c["runs_main"], *c["runs_sensitivity"]]
    sh = {n: {name: shares(toks[name], n, wset, group_of, group_names) for name in names} for n in ns}
    # the statistics and the permutations, before any figure
    pc = c["permutation"]
    tests = []
    for j, b in enumerate(pc["blocks"]):
        n, bl = b["runs"], b["block_runs"]
        series = {"share": np.array([sh[n][name]["share"] for name in names]),
                  **{g: np.array([sh[n][name]["groups"][g] for name in names]) for g in group_names}}
        res = {"runs": n, "block_runs": bl, "role": b["role"], "seed": pc["seed"] + j}
        for key, x in series.items():
            obs, dep = statistic(x)
            ref = permute(x, bl, pc["n_permutations"], pc["seed"] + j)
            res[key] = {"observed": obs, "departures": dict(zip(names, map(float, dep))),
                        "mean_share": dict(zip(names, map(float, x.mean(axis=1)))),
                        "p95": float(np.percentile(ref, 95)), "share_of_permutations_at_or_above":
                            float(np.mean(ref >= obs))}
        if b["role"] == "reading":
            res["reading"] = ("distinguishable from the relabelling of blocks" if res["share"]["observed"]
                              > res["share"]["p95"] else "insufficient evidence under this null")
        tests.append(res)
    # everything the figures need, computed and checked before the directory is made
    sc, font = c["scale"], f["font_file"]
    ImageFont.truetype(font, f["font_size"])  # a missing font stops here, not after some figures are written
    points = {n: {name: [st.run_point(x, lex) for x in st.runs(toks[name], n)] for name in names} for n in ns}
    tiles_of = {}
    for n in ns:
        px = (f["row_width_px"] - (n - 1) * f["tile_gap_px"]) // n
        tiles_of[n] = {name: [relief.grey(relief.tile(p["V"], p["A"], lambda fl: procedural_field_separable(
            fl, px, r["seed"], octaves=relief.OCTAVES, normalize=relief.NORMALIZE), r["eps"]).H, "common", d)
            for p in points[n][name]] for name in names}
    out_dir.mkdir(parents=True)
    figs, scales = {}, {}
    for n in ns:
        allv = np.array([sh[n][name]["share"] for name in names])
        pooled = float(sum(sum(sh[n][name]["function"]) for name in names)
                       / sum(sum(sh[n][name]["tokens"]) for name in names))
        fixed = (pooled - sc["fixed_width"] / 2, pooled + sc["fixed_width"] / 2)
        pct = (float(np.percentile(allv, sc["percentile_low"])), float(np.percentile(allv, sc["percentile_high"])))
        px = (f["row_width_px"] - (n - 1) * f["tile_gap_px"]) // n
        tiles = tiles_of[n]
        scales[n] = {"pooled_share": pooled, "tile_px": px}
        for sname, (lo, hi) in (("fixed", fixed), ("percentile", pct)):
            greys = {}
            clipped = 0
            for name in names:
                g, k = strip_grey(np.array(sh[n][name]["share"]), lo, hi)
                greys[name], clipped = g, clipped + k
            scales[n][sname] = {"low": lo, "high": hi, "cells_clipped": clipped}
            for v in VARIANTS:
                if v == "none" and sname == "percentile":
                    continue
                stem = f"runs{n}_none" if v == "none" else f"runs{n}_{sname}_{v}"
                a = figure([(tiles[name], greys[name]) for name in names], f, n, v)
                figs[stem] = save(a, out_dir / f"{stem}.png", f["dpi"])
            lg = legend(lo, hi, f, font, f"share of function words, {sname} scale, {n} runs")
            lg.save(out_dir / f"runs{n}_{sname}_legend.png")
            figs[f"runs{n}_{sname}_legend"] = {"file": f"runs{n}_{sname}_legend.png",
                                               "sha256": cm.sha256(out_dir / f"runs{n}_{sname}_legend.png")}
    return {"list": {"n_words": len(words), "groups": {g: len(v) for g, v in c["groups"].items()}},
            "font_sha256": cm.sha256(Path(font)), "figures": figs, "scales": {str(k): v for k, v in scales.items()},
            "tests": tests,
            "shares": {str(n): {name: sh[n][name] for name in names} for n in ns},
            "run_points": {str(n): {name: points[n][name] for name in names} for n in ns}}


def main() -> None:
    out_json = OUT_DIR / "sketch_translations_style.json"
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} exists; refusing to overwrite")
    raw, c = cm.load_conditions(COND)
    cm.need(c, "base_conditions_sha256", str)
    raw_base, base = cm.load_conditions(BASE_COND)
    cm.same_as_head([COND, BASE_COND, *(ROOT / p for p in CODE)])
    state = cm.git_state(["works", "src", "pyproject.toml", "uv.lock", ".python-version"])
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    cm.check_generator(base["generator"])
    lex = load_lexicon(ROOT / base["lexicon"]["file"])
    if lex.sha256 != base["lexicon"]["sha256"]:
        raise SystemExit("the lexicon does not match the SHA-256 of the conditions")
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    res = run(c, base, hashlib.sha256(raw_base).hexdigest(), ROOT, lex, OUT_DIR)
    res = {"meta": {"conditions_sha256": hashlib.sha256(raw).hexdigest(), "lexicon_sha256": lex.sha256,
                    "atact_library_sha256": cm.library_sha256(), "uv_lock_sha256": cm.sha256(ROOT / "uv.lock"),
                    **cm.versions("numpy", "pillow"), **state, "started": started,
                    "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}, **res}
    with out_json.open("x") as fh:
        fh.write(json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(res["tests"], indent=1))


if __name__ == "__main__":
    main()
