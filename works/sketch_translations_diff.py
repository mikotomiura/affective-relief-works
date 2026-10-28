"""Sketch of a series of translators, version 2: the departure of each translation from the mean of the translations
at the same place in the story, as relief. Chosen after the first sketch (results/sketch_translations, 4267d46) looked
nearly alike under the common scale.

Usage:  uv run python works/sketch_translations_diff.py        (refuses to overwrite)
Conditions: works/conditions/sketch_translations_diff.json (committed before this script is run)
Input:  as works/sketch_translations.py (the texts are checked against the conditions of the first sketch)
Output: results/sketch_translations_diff/sketch_translations_diff.json and its figures (written before the record;
        if a run fails between them, the directory is removed by hand before the run is repeated)

Steps: the runs of the first sketch (same extraction, tokens and cuts). For run k of translation t, d_t,k = x_t,k minus
the mean over the translations of x_.,k, for x = V and A. The departures are scaled by one number per quantity,
c_x = max over t and k of |d_x|, so that the largest departure is 1 (the input range of the rule); the tile of run k of
translation t is the tile of the rule at (d_V / c_V, d_A / c_A), under the common scale and the seed of work 1. A
departure of zero gives the tile of (0, 0) in every panel.
As references for noise, each run's matched terms are split in two ways, by position into odd and even halves and into
the first and the second half; the standard error of the run's mean is estimated as |mean_1 - mean_2| / 2. The odd-even
split assumes independent words and is small when neighbouring words are alike; the first-second split also takes the
slow change of the scene and is large; the two are read as a lower and an upper reference. Their root mean squares over
runs and translations, times sqrt(2/3) (the root mean square of a departure from the mean of three under sampling noise
alone), are reported with the root mean square of the departures. A run with fewer than two matched terms stops the run.
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

import _common as cm  # noqa: E402
import relief  # noqa: E402
import sketch_translations as st  # noqa: E402
from atact.field import procedural_field_separable  # noqa: E402
from atact.lexicon import Lexicon, load_lexicon, match_terms, tokenize  # noqa: E402

COND = ROOT / "works" / "conditions" / "sketch_translations_diff.json"
BASE_COND = ROOT / "works" / "conditions" / "sketch_translations.json"
OUT_DIR = ROOT / "results" / "sketch_translations_diff"
CODE = ["works/sketch_translations_diff.py", "works/sketch_translations.py", "works/_common.py", "src/relief.py"]
QUANTITIES = ("V", "A")


SPLITS = ("odd_even", "first_second")


def half_se(tokens: list[str], lex: Lexicon) -> dict:
    """|mean of one half of the matched terms - mean of the other| / 2 for V and A, for both splits."""
    terms = match_terms(tokens, lex)
    if len(terms) < 2:
        raise SystemExit("a run has fewer than two matched terms; its standard error cannot be estimated")
    h = len(terms) // 2
    halves = {"odd_even": (terms[0::2], terms[1::2]), "first_second": (terms[:h], terms[h:])}
    out = {}
    for split, (a, b) in halves.items():
        for j, x in enumerate(QUANTITIES):
            ma = sum(lex.entries[t][j] for t in a) / len(a)
            mb = sum(lex.entries[t][j] for t in b) / len(b)
            out[f"{split}_{x}"] = abs(ma - mb) / 2
    return out


def departures(points: dict[str, list[dict]]) -> dict[str, dict[str, list[float]]]:
    names = list(points)
    n = len(points[names[0]])
    out = {t: {x: [] for x in QUANTITIES} for t in names}
    for k in range(n):
        for x in QUANTITIES:
            m = sum(points[t][k][x] for t in names) / len(names)
            for t in names:
                out[t][x].append(points[t][k][x] - m)
    return out


def rms(v: list[float]) -> float:
    return math.sqrt(sum(a * a for a in v) / len(v))


def run(c: dict, base: dict, base_sha256: str, root: Path, lex: Lexicon, out_dir: Path) -> dict:
    st.validate(base)
    if c["base_conditions_sha256"] != base_sha256:
        raise SystemExit("the conditions of the first sketch differ from the ones named here")
    n, r, fig = base["runs"], base["relief"], base["figure"]
    d = st.dimensions(base)
    dpmm = fig["dpi"] / st.MM_PER_INCH
    px = st.tile_px(fig)
    # every extraction is checked before the lexicon is used on any text (as in the first sketch)
    toks = {}
    for t in base["translations"]:
        text = cm.check_file(root / t["file"], t["sha256"], f"the text of {t['name']}").decode("utf-8")
        kept, dropped, line = st.extract_body(text, t["heading"])
        tokens = tokenize(" ".join(kept))
        last = hashlib.sha256(kept[-1].encode("utf-8")).hexdigest()
        if (line, st.body_sha256(kept), len(tokens), dropped, last) != (
                t["heading_line"], t["body_sha256"], t["tokens"], t["dropped_captions"], t["last_line_sha256"]):
            raise SystemExit(f"{t['name']}: the extracted body differs from the conditions")
        toks[t["name"]] = st.runs(tokens, n)
    points = {name: [st.run_point(x, lex) for x in rs] for name, rs in toks.items()}
    if any(p["unmatched"] for ps in points.values() for p in ps):
        raise SystemExit("a run has no matched term; its (0, 0) would enter the mean of the translations")
    dep = departures(points)
    scale = {x: max(abs(v) for t in dep for v in dep[t][x]) for x in QUANTITIES}
    if not all(s > 0 for s in scale.values()):
        raise SystemExit("no departure at all")
    hs = {name: [half_se(tk, lex) for tk in rs] for name, rs in toks.items()}
    noise = {x: {"rms_departure": rms([v for t in dep for v in dep[t][x]]),
                 **{f"expected_rms_departure_under_noise_{sp}": math.sqrt(2 / 3) * rms(
                     [h[f"{sp}_{x}"] for name in hs for h in hs[name]]) for sp in SPLITS}} for x in QUANTITIES}
    out_dir.mkdir(parents=True)
    rec, panels = {}, []
    for t in base["translations"]:
        name = t["name"]
        tiles = [relief.grey(relief.tile(dv / scale["V"], da / scale["A"], lambda f: procedural_field_separable(
            f, px, r["seed"], octaves=relief.OCTAVES, normalize=relief.NORMALIZE), r["eps"]).H, "common", d)
            for dv, da in zip(dep[name]["V"], dep[name]["A"])]
        canvas = st.panel(tiles, fig)
        panels.append(canvas)
        rec[name] = {"figure": st.save(canvas, out_dir / f"{name}.png", fig["dpi"]),
                     "runs": [{"V": p["V"], "A": p["A"], "Z": p["Z"], "dV": dv, "dA": da, "half_se": h}
                              for p, dv, da, h in zip(points[name], dep[name]["V"], dep[name]["A"], hs[name])]}
    gap = np.ones((panels[0].shape[0], int(round(fig["panel_gap_mm"] * dpmm))))
    side = np.concatenate([x for p in panels for x in (p, gap)][:-1], axis=1)
    return {"scale": scale, "noise": noise, "side_by_side": st.save(side, out_dir / "side_by_side.png", fig["dpi"]),
            "translations": rec}


def main() -> None:
    out_json = OUT_DIR / "sketch_translations_diff.json"
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
    print(json.dumps({k: res[k] for k in ("scale", "noise", "side_by_side")}, indent=1))


if __name__ == "__main__":
    main()
