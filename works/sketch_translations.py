"""Sketch of a series of translators: three English translations of one story by Andersen, each cut into the same
number of equal runs of tokens, one tile per run, so that tile k of every translation stands at the same place in the
story. Every tile under the common height scale and one seed, as work 1.

Usage:  uv run python works/sketch_translations.py        (refuses to overwrite)
Conditions: works/conditions/sketch_translations.json (committed before this script is run)
Input:  the three Wikisource texts (external/translations-20261028/, not in the repository) and the NRC VAD Lexicon
        v2.1 (external/, not in the repository)
Output: results/sketch_translations/sketch_translations.json and the figures results/sketch_translations/*.png
        (the figures are written before the record; if a run fails between them, the directory is removed by hand
        before the run is repeated, as for work 1)

Steps per translation: the body is the run of non-empty lines after the last line that equals the heading; a line whose
letters are all upper case is an illustration caption and is dropped and listed; the body is tokenized by the
generator (atact.lexicon.tokenize) as one text, the tokens are cut into N runs at k T // N (T tokens), and each run is
matched (atact.lexicon.match_terms) and averaged as the generator averages a sentence. A run with no matched term is
made from (0, 0) and flagged. Sentence and paragraph boundaries are not used: they differ between the translators. So
a multiword entry of the lexicon is matched only inside a run, and may be matched across the end of a sentence.
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
from atact.lexicon import Lexicon, load_lexicon, match_terms, tokenize  # noqa: E402
from atact.mapping import S0, S1  # noqa: E402

COND = ROOT / "works" / "conditions" / "sketch_translations.json"
OUT_DIR = ROOT / "results" / "sketch_translations"
MM_PER_INCH = 25.4
CODE = ["works/sketch_translations.py", "works/_common.py", "src/relief.py"]


def extract_body(text: str, heading: str) -> tuple[list[str], list[str], int]:
    """Non-empty stripped lines after the last line equal to the heading; captions (letters all upper case) dropped.
    Returns the kept lines, the dropped captions and the 1-based number of the heading line."""
    lines = text.splitlines()
    at = [k for k, line in enumerate(lines) if line.strip() == heading]
    if not at:
        raise SystemExit(f"the heading {heading!r} is not in the text")
    kept, dropped = [], []
    for line in lines[at[-1] + 1 :]:
        s = line.strip()
        if not s:
            continue
        letters = [ch for ch in s if ch.isalpha()]
        (dropped if letters and all(ch.isupper() for ch in letters) else kept).append(s)
    return kept, dropped, at[-1] + 1


def tile_px(fig: dict) -> int:
    tile_mm = (fig["panel_width_mm"] - (fig["columns"] - 1) * fig["gap_mm"]) / fig["columns"]
    return int(tile_mm * fig["dpi"] / MM_PER_INCH)


def body_sha256(kept: list[str]) -> str:
    return hashlib.sha256("\n".join(kept).encode("utf-8")).hexdigest()


def runs(tokens: list[str], n: int) -> list[list[str]]:
    """The tokens cut into n runs at k T // n."""
    t = len(tokens)
    return [tokens[k * t // n : (k + 1) * t // n] for k in range(n)]


def run_point(tokens: list[str], lex: Lexicon) -> dict:
    """The mean of the matched terms, as atact.lexicon.sentence_point does for a sentence."""
    terms = match_terms(tokens, lex)
    if not terms:
        return {"V": 0.0, "A": 0.0, "Z": 0, "tokens": len(tokens), "unmatched": True}
    z = len(terms)
    return {"V": sum(lex.entries[x][0] for x in terms) / z, "A": sum(lex.entries[x][1] for x in terms) / z, "Z": z,
            "tokens": len(tokens), "unmatched": False}


def validate(c: dict) -> None:
    n = cm.need(c, "runs", int)
    if n < 2:
        raise SystemExit("at least two runs")
    names = [cm.need(t, "name", str, "translations.") for t in c["translations"]]
    if len(names) != len(set(names)) or len(names) < 2:
        raise SystemExit("the translations need distinct names, at least two")
    for t in c["translations"]:
        for k in ("file", "sha256", "heading", "body_sha256"):
            cm.need(t, k, str, "translations.")
        for k in ("heading_line", "tokens"):
            cm.need(t, k, int, "translations.")
        cm.need(t, "last_line_sha256", str, "translations.")
        if not all(isinstance(x, str) for x in cm.need(t, "dropped_captions", list, "translations.")):
            raise SystemExit(f"{t['name']}: the dropped captions are listed as strings")
        if t["tokens"] < 10 * n:
            raise SystemExit(f"{t['name']}: fewer than ten tokens a run")
    r = c["relief"]
    d = dimensions(c)
    if d.s_max < S0 + S1:
        raise SystemExit("s_max is below the largest s of the rule; the common scale would leave the span D")
    if cm.need(r, "G_star", float, "relief.") != relief.G_STAR or not math.isclose(
            cm.need(r, "eps", float, "relief."), relief.eps_for(d, r["G_star"]), rel_tol=1e-12):
        raise SystemExit("eps is not e_min G* / (2 L) with the G* of the paper on the rule")
    if (cm.need(r, "octaves", int, "relief.") != relief.OCTAVES or r.get("normalize") != relief.NORMALIZE
            or cm.need(r, "seed", int, "relief.") < 0):
        raise SystemExit("the field settings differ from those of src/relief.py")
    f = c["figure"]
    for k in ("panel_width_mm", "gap_mm", "panel_gap_mm"):
        cm.need(f, k, float, "figure.")
    cols, dpi = cm.need(f, "columns", int, "figure."), cm.need(f, "dpi", int, "figure.")
    if cols < 1 or dpi < 1:
        raise SystemExit("columns and dpi are positive")


def dimensions(c: dict) -> relief.Dimensions:
    r = c["relief"]
    return relief.Dimensions(*(cm.need(r, k, float, "relief.") for k in ("L_mm", "D_mm", "s_max", "base_mm", "e_min_mm")))


def panel(tiles: list[np.ndarray], fig: dict) -> np.ndarray:
    """One translation: the tiles in reading order, fig['columns'] to a row, origin of each tile lower left."""
    cols = fig["columns"]
    rows = math.ceil(len(tiles) / cols)
    dpmm = fig["dpi"] / MM_PER_INCH
    tile_mm = (fig["panel_width_mm"] - (cols - 1) * fig["gap_mm"]) / cols
    px = tile_px(fig)
    step = (tile_mm + fig["gap_mm"]) * dpmm
    h = int(round((rows * tile_mm + (rows - 1) * fig["gap_mm"]) * dpmm))
    canvas = np.ones((h, int(round(fig["panel_width_mm"] * dpmm))))
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        y, x = int(round(r * step)), int(round(c * step))
        canvas[y : y + px, x : x + px] = np.flipud(np.clip(t, 0.0, 1.0))
    return canvas


def save(a: np.ndarray, path: Path, dpi: int) -> dict:
    Image.fromarray(np.round(a * 255).astype(np.uint8), mode="L").save(path, dpi=(dpi, dpi))
    return {"file": path.name, "sha256": cm.sha256(path)}


def ranks(v: list[float]) -> np.ndarray:
    """Ranks from 0, tied values given the mean of their ranks."""
    a = np.asarray(v, dtype=float)
    order = np.argsort(a, kind="stable")
    r = np.empty(len(a))
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        r[order[i : j + 1]] = (i + j) / 2
        i = j + 1
    return r


def spearman(x: list[float], y: list[float]) -> float:
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def run(c: dict, root: Path, lex: Lexicon, out_dir: Path) -> dict:
    """The sketch from the conditions; the inputs are read from root. Writes the figures, returns the record."""
    validate(c)
    n, r, fig = c["runs"], c["relief"], c["figure"]
    d = dimensions(c)
    dpmm = fig["dpi"] / MM_PER_INCH
    px = tile_px(fig)
    # every extraction is checked before the lexicon is used on any text
    bodies = {}
    for t in c["translations"]:
        text = cm.check_file(root / t["file"], t["sha256"], f"the text of {t['name']}").decode("utf-8")
        kept, dropped, line = extract_body(text, t["heading"])
        tokens = tokenize(" ".join(kept))
        last = hashlib.sha256(kept[-1].encode("utf-8")).hexdigest()
        if (line, body_sha256(kept), len(tokens), dropped, last) != (
                t["heading_line"], t["body_sha256"], t["tokens"], t["dropped_captions"], t["last_line_sha256"]):
            raise SystemExit(f"{t['name']}: the extracted body differs from the conditions")
        bodies[t["name"]] = (tokens, dropped)
    out_dir.mkdir(parents=True)
    rec, panels = {}, []
    for t in c["translations"]:
        tokens, dropped = bodies[t["name"]]
        points = [run_point(x, lex) for x in runs(tokens, n)]
        tiles = [relief.grey(relief.tile(p["V"], p["A"], lambda f: procedural_field_separable(
            f, px, r["seed"], octaves=relief.OCTAVES, normalize=relief.NORMALIZE), r["eps"]).H, "common", d)
            for p in points]
        canvas = panel(tiles, fig)
        panels.append(canvas)
        rec[t["name"]] = {"dropped_captions": dropped, "figure": save(canvas, out_dir / f"{t['name']}.png", fig["dpi"]),
                          "unmatched": sum(p["unmatched"] for p in points),
                          "Z_min": min(p["Z"] for p in points), "Z_median": float(np.median([p["Z"] for p in points])),
                          "runs": points}
    gap = np.ones((panels[0].shape[0], int(round(fig["panel_gap_mm"] * dpmm))))
    side = np.concatenate([x for p in panels for x in (p, gap)][:-1], axis=1)
    names = [t["name"] for t in c["translations"]]
    pairs = {f"{a}~{b}": {k: spearman([p[k] for p in rec[a]["runs"]], [p[k] for p in rec[b]["runs"]]) for k in ("V", "A")}
             for i, a in enumerate(names) for b in names[i + 1 :]}
    return {"tile_px": px, "side_by_side": save(side, out_dir / "side_by_side.png", fig["dpi"]),
            "spearman_by_position": pairs, "translations": rec}


def main() -> None:
    out_json = OUT_DIR / "sketch_translations.json"
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} exists; refusing to overwrite")
    raw, c = cm.load_conditions(COND)
    validate(c)
    cm.same_as_head([COND, *(ROOT / p for p in CODE)])
    state = cm.git_state(["works", "src", "pyproject.toml", "uv.lock", ".python-version"])
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    cm.check_generator(c["generator"])
    lex = load_lexicon(ROOT / c["lexicon"]["file"])
    if lex.sha256 != c["lexicon"]["sha256"]:
        raise SystemExit("the lexicon does not match the SHA-256 of the conditions")
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    res = run(c, ROOT, lex, OUT_DIR)
    res = {"meta": {"conditions_sha256": hashlib.sha256(raw).hexdigest(), "lexicon_sha256": lex.sha256,
                    "atact_library_sha256": cm.library_sha256(), "uv_lock_sha256": cm.sha256(ROOT / "uv.lock"),
                    **cm.versions("numpy", "pillow"), **state, "started": started,
                    "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}, **res}
    with out_json.open("x") as fh:
        fh.write(json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({k: res[k] for k in ("side_by_side", "spearman_by_position")}, indent=1))


if __name__ == "__main__":
    main()
