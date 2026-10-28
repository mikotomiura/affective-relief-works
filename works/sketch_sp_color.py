"""Work 2, the refined sketches: the relief SP (percentiles among the words of the works instead of map B, a seed per
word or spelling) and the colour C1 in OKLCH over the lightness of the relief, with the percentiles alone and the seed
alone, for the rows of the check of the seeds under two common seeds.

Usage:  uv run python works/sketch_sp_color.py        (refuses to overwrite)
Conditions: works/conditions/sketch_sp_color.json (committed before this script was run)
Input:  the 23 works of work 2 (as works/work2_words.py), the lexicon of work 2, results/work2_words/words.json,
        results/work2_variants/variants.json
Output: results/sketch_sp_color/
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.pycache_prefix = tempfile.mkdtemp(prefix="pycache-")  # never run a stale .pyc of src/ (ignored by git)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "works"))
sys.path.insert(0, str(ROOT / "checks"))

import numpy as np  # noqa: E402

import _common as cm  # noqa: E402
import japanese_text  # noqa: E402
import relief  # noqa: E402
from atact.mapping import params  # noqa: E402
from pipeline import calibration_map, read_aozora_zip, work2_body, work2_text  # noqa: E402
from sketch_color_features import figure, grey_u8, spelling_seed  # noqa: E402
from work2_words import z_mm  # noqa: E402

COND = ROOT / "works" / "conditions" / "sketch_sp_color.json"
WORDS_COND = ROOT / "works" / "conditions" / "work2_words.json"
VARIANTS_COND = ROOT / "works" / "conditions" / "work2_variants.json"
WORK2_COND = ROOT / "works" / "conditions" / "work2.json"
WORK2_RESULT = ROOT / "results" / "work2" / "work2.json"
OUT_DIR = ROOT / "results" / "sketch_sp_color"
CODE = ["works/sketch_sp_color.py", "works/sketch_color_features.py", "works/work2_words.py", "works/_common.py",
        "src/pipeline.py", "src/japanese_text.py", "src/relief.py"]
KNOWN = ("G0", "P", "S", "SP", "C1", "SPC")
HUE0, HUE_SPAN = 260.0, 155.0
C0, C_SPAN, C_STEP = 0.03, 0.12, 0.002


# ---------------------------------------------------------------- percentiles


def weighted_percentile(values: np.ndarray, weights: np.ndarray, x: float) -> float:
    """Weight of the values below x plus half the weight of those equal to x, over the total weight."""
    total = weights.sum()
    if not total > 0:
        raise ValueError("no weight")
    return float((weights[values < x].sum() + 0.5 * weights[values == x].sum()) / total)


def reference_weights(counts: dict[str, Counter]) -> dict[str, dict[str, float]]:
    """Per word, three weights: per author equal (each author's tokens sum to 1), tokens, types."""
    out: dict[str, dict[str, float]] = {}
    for a, cnt in counts.items():
        T = sum(cnt.values())
        if T == 0:
            raise ValueError(f"{a}: no token")
        for w, n in cnt.items():
            d = out.setdefault(w, {"author_equal": 0.0, "tokens": 0.0, "types": 1.0})
            d["author_equal"] += n / T
            d["tokens"] += n
    return out


# ---------------------------------------------------------------- colour (OKLab, Ottosson 2020)


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
               [0.0883024619, 0.2817188376, 0.6299787005]])
M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
               [0.0259040371, 0.7827717662, -0.8086757660]])


def linear_to_oklab(rgb: np.ndarray) -> np.ndarray:
    return np.cbrt(rgb @ M1.T) @ M2.T


def oklab_to_linear(lab: np.ndarray) -> np.ndarray:
    return (np.linalg.solve(M2, lab.reshape(-1, 3).T).T ** 3 @ np.linalg.inv(M1).T).reshape(lab.shape)


def oklch_tint(grey: np.ndarray, hue_deg: float, chroma: float) -> tuple[np.ndarray, float]:
    """RGB with the OKLab lightness of the grey, the hue given and the largest chroma <= chroma that lies in sRGB;
    also the share of pixels whose chroma was reduced."""
    g = srgb_to_linear(grey.astype(float) / 255.0)
    L = linear_to_oklab(np.stack([g, g, g], axis=-1))[..., 0]
    h = np.radians(hue_deg % 360.0)
    C = np.full(L.shape, chroma)
    reduced = np.zeros(L.shape, dtype=bool)
    while True:
        lab = np.stack([L, C * np.cos(h), C * np.sin(h)], axis=-1)
        rgb = oklab_to_linear(lab)
        out = ((rgb < -1e-6) | (rgb > 1 + 1e-6)).any(axis=-1) & (C > 0)
        if not out.any():
            break
        reduced |= out
        C = np.where(out, np.maximum(C - C_STEP, 0.0), C)
    return np.round(linear_to_srgb(rgb) * 255).astype(np.uint8), float(reduced.mean())


def hue_of(p_valence: float) -> float:
    return HUE0 + HUE_SPAN * p_valence


def chroma_of(p_arousal: float) -> float:
    return C0 + C_SPAN * p_arousal


# ---------------------------------------------------------------- main


def main() -> None:
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} exists; refusing to overwrite")
    raw = COND.read_bytes()
    c = json.loads(raw)
    if tuple(c["candidates"]) != KNOWN:
        raise SystemExit(f"unknown or missing candidates: {list(c['candidates'])}")
    _, w2 = cm.load_conditions(WORK2_COND)
    if c["inputs"]["work2_conditions"] != WORK2_COND.relative_to(ROOT).as_posix():
        raise SystemExit("the conditions name another work 2")
    words_path, variants_path = ROOT / c["inputs"]["words"], ROOT / c["inputs"]["variants"]
    cm.same_as_head([COND, WORK2_COND, WORDS_COND, VARIANTS_COND, WORK2_RESULT, words_path, variants_path,
                     ROOT / w2["map"]["file"], *(ROOT / p for p in CODE)])
    state = cm.git_state(["works", "src", "checks", "results/work2", "results/work2_words", "results/work2_variants",
                          "results/calibrate_memolon.json", "pyproject.toml", "uv.lock"])
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    wc = json.loads(WORDS_COND.read_bytes())
    if wc["work2_conditions_sha256"] != cm.sha256(WORK2_COND):
        raise SystemExit("works/conditions/work2.json differs from the one the words were selected on")
    cm.check_generator(wc["generator"])
    cm.check_versions(w2["tagger"], ("fugashi", "ipadic"))
    tagger = japanese_text.Tagger()
    if tagger.dicdir_sha256 != w2["tagger"]["dicdir_sha256"]:
        raise SystemExit("the files of the MeCab dictionary differ from those of work 2")
    lex = japanese_text.load_memolon(ROOT / w2["lexicon"]["file"])
    if lex.sha256 != w2["lexicon"]["sha256"]:
        raise SystemExit("the lexicon differs from that of work 2")
    vmap = calibration_map(cm.check_file(ROOT / w2["map"]["file"], w2["map"]["sha256"], "the calibration"))
    words = json.loads(words_path.read_bytes())
    variants = json.loads(variants_path.read_bytes())
    if (words["meta"]["conditions_sha256"] != cm.sha256(WORDS_COND)
            or variants["meta"]["conditions_sha256"] != cm.sha256(VARIANTS_COND)):
        raise SystemExit("a result was not computed on the conditions named")
    eps = relief.eps_for(relief.Dimensions(**{k: wc["relief"][k] for k in ("L_mm", "D_mm", "s_max", "base_mm",
                                                                         "e_min_mm")}))
    if eps != words["eps"] or eps != variants["eps"]:
        raise SystemExit("eps differs from the records")
    D = wc["relief"]["D_mm"]
    authors, seeds, rows = c["authors"], c["seeds"], c["rows"]
    if authors != wc["trial_authors"]:
        raise SystemExit("the authors differ from the trial authors")
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    # the reference: matched tokens of the 23 works, per author
    done = {w["work_id"]: w for ws in json.loads(WORK2_RESULT.read_text())["authors"].values() for w in ws}
    counts: dict[str, Counter] = {}
    for w in w2["works"]:
        cm.check_file(ROOT / w["zip"], w["zip_sha256"], f"the zip of {w['work_id']}")
        name, data = read_aozora_zip(ROOT / w["zip"])
        if name != w["txt_name"]:
            raise SystemExit(f"{w['work_id']}: the zip holds {name}")
        text = data.decode(w["encoding"])
        body, _, _, kept, _ = work2_body(text)
        if hashlib.sha256(body.encode("utf-8")).hexdigest() != w["body_sha256"] or len(kept) != w["sentences"]:
            raise SystemExit(f"{w['work_id']}: the body differs from work 2")
        t = work2_text(text, lex, tagger, vmap)
        ref = done[w["work_id"]]
        if len(t.points) != ref["sentences"] or sum(p.Z for p in t.points) != ref["matched_tokens"]:
            raise SystemExit(f"{w['work_id']}: the text step gives another result than results/work2")
        counts.setdefault(w["author"], Counter()).update(x for p in t.points for x in p.matched)
    if len(counts) != 6:
        raise SystemExit(f"expected six authors, got {len(counts)}")
    wts = reference_weights(counts)
    ref_words = sorted(wts)
    vals = {k: np.array([lex.entries[w][i] for w in ref_words]) for k, i in (("valence", 0), ("arousal", 1))}
    weights = {m: np.array([wts[w][m] for w in ref_words]) for m in ("author_equal", "tokens", "types")}
    all_v = np.sort(np.array([v for v, _ in lex.entries.values()]))
    all_a = np.sort(np.array([a for _, a in lex.entries.values()]))

    # the keys drawn: spellings of the sets and the 20 words, with the entry that gave their values
    sets = [r for r in rows if r != "words20"]
    entry, recorded = {}, {}
    for k, v in variants["values"].items():
        if v["set"] in sets:
            if [float(x) for x in params(v["V"], v["A"])] != [v["f"], v["s"], v["gamma"]]:
                raise SystemExit(f"{k}: params differ from the record")
            entry[k], recorded[k] = v["entry"], (v["V"], v["A"])
    for a in authors:
        for w in words["authors"][a]["words"]:
            if [float(x) for x in params(w["V"], w["A"])] != [w["f"], w["s"], w["gamma"]]:
                raise SystemExit(f"{w['word']}: params differ from the record")
            if w["word"] in recorded and recorded[w["word"]] != (w["V"], w["A"]):
                raise SystemExit(f"{w['word']}: two recorded values")
            entry[w["word"]], recorded[w["word"]] = w["word"], (w["V"], w["A"])
    values = {}
    for k, e in entry.items():
        if e not in wts:
            raise SystemExit(f"{k} ({e}) does not occur in the reference")
        if tuple(float(x) for x in vmap(*lex.entries[e])) != recorded[k]:
            raise SystemExit(f"{k}: the entry {e} does not give the recorded V and A")
        rv, ra = lex.entries[e]
        values[k] = {"entry": e, "count": wts[e]["tokens"], "V": recorded[k][0], "A": recorded[k][1],
                     **{f"p_{dim}_{m}": weighted_percentile(vals[dim], weights[m], x)
                        for dim, x in (("valence", rv), ("arousal", ra)) for m in weights},
                     "p_valence_lexicon": float((np.searchsorted(all_v, rv) + np.searchsorted(all_v, rv, "right")) / 2 / len(all_v)),
                     "p_arousal_lexicon": float((np.searchsorted(all_a, ra) + np.searchsorted(all_a, ra, "right")) / 2 / len(all_a)),
                     "seed_SP": {str(s): spelling_seed(k, s) for s in seeds}}

    def seq(row: str, a: str) -> list[str]:
        if row == "words20":
            return [w["word"] for w in words["authors"][a]["words"]]
        t = variants["rows"][row][a]["tiles"]
        return [k for k, n in t.items() for _ in range(n)]

    px, k = c["drawing"]["tile_px"], c["drawing"]["supersample"]
    cache: dict[tuple, np.ndarray] = {}
    reduced_share: dict[str, list[float]] = {"C1": [], "SPC": []}

    def relief_u8(key: str, seed: int, use_pct: bool, use_seed: bool, fresh: bool = False) -> np.ndarray:
        ck = (key, seed, use_pct, use_seed)
        if fresh or ck not in cache:
            v = values[key]
            V, A = ((2 * v["p_valence_author_equal"] - 1, 2 * v["p_arousal_author_equal"] - 1) if use_pct
                    else (v["V"], v["A"]))
            z = z_mm(V, A, spelling_seed(key, seed) if use_seed else seed, px * k, wc, eps)
            g = grey_u8(z.reshape(px, k, px, k).mean(axis=(1, 3)), D)
            if fresh:
                return g
            cache[ck] = g
        return cache[ck]

    def tile(cand: str, key: str, seed: int, fresh: bool = False) -> np.ndarray:
        use_pct, use_seed = cand in ("P", "SP", "SPC"), cand in ("S", "SP", "SPC")
        g = relief_u8(key, seed, use_pct, use_seed, fresh)
        if cand in ("C1", "SPC"):
            v = values[key]
            rgb, share = oklch_tint(g, hue_of(v["p_valence_author_equal"]), chroma_of(v["p_arousal_author_equal"]))
            if not fresh:
                reduced_share[cand].append(share)
            return rgb
        return np.repeat(g[..., None], 3, axis=-1)

    k3 = {}
    for row in rows:
        for a in authors:
            q = seq(row, a)
            k3[f"{row}/{a}"] = {
                m: {"range": float(np.ptp(x)), "iqr": float(np.subtract(*np.percentile(x, [75, 25])))}
                for m, x in (("p_valence_author_equal", [values[w]["p_valence_author_equal"] for w in q]),
                             ("p_arousal_author_equal", [values[w]["p_arousal_author_equal"] for w in q]),
                             ("p_valence_lexicon", [values[w]["p_valence_lexicon"] for w in q]),
                             ("p_arousal_lexicon", [values[w]["p_arousal_lexicon"] for w in q]))}

    tmp = Path(tempfile.mkdtemp(prefix=".sketch_sp_color-", dir=OUT_DIR.parent))
    try:
        figs, k0, n_keys = {}, {}, {}
        font = wc["tiles"]["font_file"]
        for cand in KNOWN:
            blocks, same = [], True
            for row in rows:
                lines = []
                for s in seeds:
                    drawn = {a: [tile(cand, key, s) for key in seq(row, a)] for a in authors}
                    lines += [(f"{words['authors'][a]['slug']} {s}", drawn[a]) for a in authors]
                    for key in sorted(set(seq(row, authors[0])) & set(seq(row, authors[1]))):  # K0
                        fresh = tile(cand, key, s, fresh=True)
                        for a in authors:
                            for i in (i for i, x in enumerate(seq(row, a)) if x == key):
                                same &= bool(np.array_equal(drawn[a][i], fresh))
                        n_keys[cand] = n_keys.get(cand, 0) + 1
                blocks.append((f"{cand}  {row}", lines))
            k0[cand] = same
            name = f"sketch_{cand}.png"
            figure(blocks, font, px).save(tmp / name)
            figs[name] = cm.sha256(tmp / name)
        if not all(k0.values()):
            raise SystemExit(f"K0 fails: {k0}")
        with (tmp / "values.json").open("x") as fh:
            fh.write(json.dumps({"reference_words": len(ref_words),
                                 "reference_tokens": {a: sum(n.values()) for a, n in counts.items()},
                                 "values": values}, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
        res = {"meta": {"conditions_sha256": hashlib.sha256(raw).hexdigest(), "script_sha256": cm.sha256(Path(__file__)),
                        "font_sha256": cm.sha256(Path(font)), "lexicon_sha256": lex.sha256, "dicdir_sha256": tagger.dicdir_sha256,
                        **cm.versions("numpy", "scipy", "pillow", "fugashi", "ipadic"), **state,
                        "started": started, "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
               "eps": eps, "K0_note": "the tile of a shared key drawn again without any cache equals every occurrence in both rows; tile() takes no author, so this checks that drawing is repeatable, the author-free construction is by the code",
               "K3_note": "the spread of the author-equal percentiles (the ones drawn) and of the whole-lexicon ones; the token and type percentiles are in values.json",
               "K0_same_key_same_tile": k0, "K0_keys_compared": n_keys,
               "K2b_share_of_pixels_with_chroma_reduced": {m: float(np.mean(x)) for m, x in reduced_share.items()},
               "K3_spread_of_percentiles": k3, "figures": figs, "values_sha256": cm.sha256(tmp / "values.json")}
        with (tmp / "sketch.json").open("x") as fh:
            fh.write(json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    except BaseException:
        shutil.rmtree(tmp)
        raise
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} appeared during the run; the output is left in {tmp}")
    tmp.rename(OUT_DIR)
    print(json.dumps({"K0": k0, "K2b": res["K2b_share_of_pixels_with_chroma_reduced"]}))


if __name__ == "__main__":
    main()
