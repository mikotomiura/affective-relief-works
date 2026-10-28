"""Work 2, sketches of colour and of features outside the rule: the rows of the check of the seeds drawn again under
two common seeds with each candidate of the conditions (grey, colour from the values, colour per spelling, colour from
the basic emotions, seed from the spelling, percentiles instead of map B, and combinations).

Usage:  uv run python works/sketch_color_features.py        (refuses to overwrite)
Conditions: works/conditions/sketch_color_features.json (committed before this script was run)
Input:  results/work2_words/words.json, results/work2_variants/variants.json, the Japanese lexicon (outside git)
Output: results/sketch_color_features/
"""

from __future__ import annotations

import colorsys
import csv
import datetime as dt
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.pycache_prefix = tempfile.mkdtemp(prefix="pycache-")  # never run a stale .pyc of src/ (ignored by git)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "works"))
sys.path.insert(0, str(ROOT / "checks"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import _common as cm  # noqa: E402
import relief  # noqa: E402
from atact.mapping import params  # noqa: E402
from work2_words import z_mm  # noqa: E402

COND = ROOT / "works" / "conditions" / "sketch_color_features.json"
WORDS_COND = ROOT / "works" / "conditions" / "work2_words.json"
VARIANTS_COND = ROOT / "works" / "conditions" / "work2_variants.json"
OUT_DIR = ROOT / "results" / "sketch_color_features"
CODE = ["works/sketch_color_features.py", "works/work2_words.py", "works/_common.py", "src/relief.py"]
EMOTIONS = ("joy", "anger", "sadness", "fear", "disgust")
EMOTION_HUE = {"joy": 50, "anger": 0, "sadness": 220, "fear": 280, "disgust": 120}
PALETTE = (0, 210, 120, 45, 285, 175, 330, 90)
COLUMNS = ("valence", "arousal", "dominance", *EMOTIONS)
KNOWN = ("G0", "C1", "C2", "E5", "S1", "P1", "SP", "SPC")


# ---------------------------------------------------------------- features


def percentile(sorted_values: np.ndarray, x: float) -> float:
    """Fraction of values smaller than x plus half of those equal to x."""
    lo = np.searchsorted(sorted_values, x, side="left")
    hi = np.searchsorted(sorted_values, x, side="right")
    return float((lo + 0.5 * (hi - lo)) / len(sorted_values))


def spelling_seed(spelling: str, common: int) -> int:
    return (int.from_bytes(hashlib.sha256(spelling.encode("utf-8")).digest()[:8], "big") + common) % 2**31


def hue_from_valence(p: float) -> float:
    return 220.0 + (25.0 - 220.0) * p


def sat_from_arousal(p: float) -> float:
    return 0.15 + 0.70 * p


def emotion_colour(z: dict[str, float]) -> tuple[float, float]:
    top = max(EMOTIONS, key=lambda e: (z[e], -EMOTIONS.index(e)))
    return float(EMOTION_HUE[top]), 0.05 + 0.80 * min(max(z[top], 0.0), 3.0) / 3.0


def tint(grey_u8: np.ndarray, hue: float | None, sat: float) -> np.ndarray:
    """RGB with the lightness of the grey and the hue and saturation given (hue None: grey)."""
    g = grey_u8.astype(float) / 255.0
    if hue is None:
        return np.repeat(grey_u8[..., None], 3, axis=-1)
    h = (hue % 360.0) / 360.0
    rgb = np.vectorize(lambda l: colorsys.hls_to_rgb(h, l, sat), otypes=[float, float, float])(g)
    return np.round(np.stack(rgb, axis=-1) * 255).astype(np.uint8)


def grey_u8(z: np.ndarray, D: float) -> np.ndarray:
    return np.round(np.flipud(np.clip(z / D + 0.5, 0.0, 1.0)) * 255).astype(np.uint8)


# ---------------------------------------------------------------- lexicon


def read_lexicon(path: Path, wanted: set[str]) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    """Sorted columns over all entries and the raw values of the wanted entries."""
    cols: dict[str, list[float]] = {k: [] for k in COLUMNS}
    rows: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        r = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        if tuple(next(r)) != ("word", *COLUMNS):
            raise SystemExit("unexpected columns of the lexicon")
        for row in r:
            if len(row) != len(COLUMNS) + 1:
                raise SystemExit(f"a row of the lexicon has {len(row)} fields")
            v = [float(x) for x in row[1:]]
            if not all(np.isfinite(v)):
                raise SystemExit(f"{row[0]}: a value is not finite")
            for k, x in zip(COLUMNS, v):
                cols[k].append(x)
            if row[0] in wanted:
                if row[0] in rows:
                    raise SystemExit(f"{row[0]} occurs twice in the lexicon")
                rows[row[0]] = dict(zip(COLUMNS, v))
    missing = wanted - rows.keys()
    if missing:
        raise SystemExit(f"not in the lexicon: {sorted(missing)}")
    return {k: np.sort(np.array(v)) for k, v in cols.items()}, rows


# ---------------------------------------------------------------- figure


def figure(blocks: list[tuple[str, list[tuple[str, list[np.ndarray]]]]], font_path: str, px: int) -> Image.Image:
    gap, lab, bgap = 3, 150, 18
    n = max(len(t) for _, lines in blocks for _, t in lines)
    W = lab + n * (px + gap)
    H = sum(22 + len(lines) * (px + gap) + bgap for _, lines in blocks)
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, 16)
    y = 0
    for title, lines in blocks:
        d.text((0, y), title, fill=(0, 0, 0), font=font)
        y += 22
        for label, tiles in lines:
            d.text((0, y + px // 2 - 8), label, fill=(0, 0, 0), font=font)
            for k, t in enumerate(tiles):
                img.paste(Image.fromarray(t, mode="RGB"), (lab + k * (px + gap), y))
            y += px + gap
        y += bgap
    return img


# ---------------------------------------------------------------- main


def main() -> None:
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} exists; refusing to overwrite")
    raw = COND.read_bytes()
    c = json.loads(raw)
    inp = c["inputs"]
    lex_path = ROOT / inp["lexicon"]
    if cm.sha256(lex_path) != inp["lexicon_sha256"]:
        raise SystemExit("the lexicon differs from the one named in the conditions")
    words_path, variants_path = ROOT / inp["words"], ROOT / inp["variants"]
    cm.same_as_head([COND, WORDS_COND, VARIANTS_COND, words_path, variants_path, *(ROOT / p for p in CODE)])
    state = cm.git_state(["works", "src", "results/work2_words", "results/work2_variants", "pyproject.toml", "uv.lock"])
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    wc = json.loads(WORDS_COND.read_bytes())
    cm.check_generator(wc["generator"])
    words = json.loads(words_path.read_bytes())
    variants = json.loads(variants_path.read_bytes())
    if (words["meta"]["conditions_sha256"] != cm.sha256(WORDS_COND)
            or variants["meta"]["conditions_sha256"] != cm.sha256(VARIANTS_COND)):
        raise SystemExit("a result was not computed on the conditions named")
    if tuple(c["candidates"]) != KNOWN:
        raise SystemExit(f"unknown or missing candidates: {list(c['candidates'])}")
    eps = relief.eps_for(relief.Dimensions(**{k: wc["relief"][k] for k in ("L_mm", "D_mm", "s_max", "base_mm",
                                                                         "e_min_mm")}))
    if eps != words["eps"] or eps != variants["eps"]:
        raise SystemExit("eps differs from the records")
    D = wc["relief"]["D_mm"]
    authors, seeds, rows = c["authors"], c["seeds"], c["rows"]
    if authors != wc["trial_authors"]:
        raise SystemExit("the authors differ from the trial authors")
    slug = {a: words["authors"][a]["slug"] for a in authors}
    sets = [r for r in rows if r != "words20"]

    def seq(row: str, a: str) -> list[str]:
        if row == "words20":
            return [w["word"] for w in words["authors"][a]["words"]]
        t = variants["rows"][row][a]["tiles"]
        return [k for k, n in t.items() for _ in range(n)]

    recorded = {k: (v["V"], v["A"]) for k, v in variants["values"].items() if v["set"] in sets}
    entry = {k: v["entry"] for k, v in variants["values"].items() if v["set"] in sets}
    for k, v in variants["values"].items():
        if v["set"] in sets and [float(x) for x in params(v["V"], v["A"])] != [v["f"], v["s"], v["gamma"]]:
            raise SystemExit(f"{k}: params differ from the record")
    for a in authors:
        for w in words["authors"][a]["words"]:
            if [float(x) for x in params(w["V"], w["A"])] != [w["f"], w["s"], w["gamma"]]:
                raise SystemExit(f"{w['word']}: params differ from the record")
            if w["word"] in recorded and recorded[w["word"]] != (w["V"], w["A"]):
                raise SystemExit(f"{w['word']}: two recorded values")
            recorded[w["word"]] = (w["V"], w["A"])
            entry.setdefault(w["word"], w["word"])
    sorted_cols, lex_by_entry = read_lexicon(lex_path, set(entry.values()))
    lex = {k: lex_by_entry[entry[k]] for k in recorded}
    mb = c["map_b"]
    for k, (V, A) in recorded.items():  # the lookup gives the recorded values back through map B
        vb = min(max(mb["V"][0] * lex[k]["valence"] + mb["V"][1], -1.0), 1.0)
        ab = min(max(mb["A"][0] * lex[k]["arousal"] + mb["A"][1], -1.0), 1.0)
        if abs(vb - V) > mb["tolerance"] or abs(ab - A) > mb["tolerance"]:
            raise SystemExit(f"{k}: the lexicon entry {entry[k]} does not give the recorded V and A through map B")
    mean = {e: float(sorted_cols[e].mean()) for e in EMOTIONS}
    sd = {e: float(sorted_cols[e].std()) for e in EMOTIONS}
    values = {}
    for k, (V, A) in recorded.items():
        pv, pa = percentile(sorted_cols["valence"], lex[k]["valence"]), percentile(sorted_cols["arousal"], lex[k]["arousal"])
        values[k] = {"V": V, "A": A, "entry": entry[k], "raw": lex[k], "p_valence": pv, "p_arousal": pa,
                     "p_dominance": percentile(sorted_cols["dominance"], lex[k]["dominance"]),
                     "z": {e: (lex[k][e] - mean[e]) / sd[e] for e in EMOTIONS},
                     "seed_S1": {str(s): spelling_seed(k, s) for s in seeds}}
    palette_of = {}
    for s in sets:
        keys = [k for k, v in variants["values"].items() if v["set"] == s]
        if len(keys) > len(PALETTE):
            raise SystemExit(f"{s}: more spellings than colours")
        palette_of.update({k: PALETTE[i] for i, k in enumerate(keys)})

    px, k = c["drawing"]["tile_px"], c["drawing"]["supersample"]
    cache: dict[tuple, np.ndarray] = {}

    def relief_u8(key: str, seed: int, use_seed: bool, use_pct: bool) -> np.ndarray:
        ck = (key, seed, use_seed, use_pct)
        if ck not in cache:
            v = values[key]
            V, A = (2 * v["p_valence"] - 1, 2 * v["p_arousal"] - 1) if use_pct else (v["V"], v["A"])
            sd_ = spelling_seed(key, seed) if use_seed else seed
            z = z_mm(V, A, sd_, px * k, wc, eps).reshape(px, k, px, k).mean(axis=(1, 3))
            cache[ck] = grey_u8(z, D)
        return cache[ck]

    def tile(cand: str, key: str, seed: int, row: str, fresh: bool = False) -> np.ndarray:
        use_seed, use_pct = cand in ("S1", "SP", "SPC"), cand in ("P1", "SP", "SPC")
        if fresh:
            v = values[key]
            V, A = (2 * v["p_valence"] - 1, 2 * v["p_arousal"] - 1) if use_pct else (v["V"], v["A"])
            z = z_mm(V, A, spelling_seed(key, seed) if use_seed else seed, px * k, wc, eps)
            g = grey_u8(z.reshape(px, k, px, k).mean(axis=(1, 3)), D)
        else:
            g = relief_u8(key, seed, use_seed, use_pct)
        v = values[key]
        if cand in ("C1", "SPC"):
            return tint(g, hue_from_valence(v["p_valence"]), sat_from_arousal(v["p_arousal"]))
        if cand == "C2":
            return tint(g, palette_of[key], 0.7) if row != "words20" else tint(g, None, 0)
        if cand == "E5":
            return tint(g, *emotion_colour(v["z"]))
        return tint(g, None, 0)

    def tile_fresh(cand: str, key: str, seed: int, row: str) -> np.ndarray:
        return tile(cand, key, seed, row, fresh=True)

    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    tmp = Path(tempfile.mkdtemp(prefix=".sketch_color_features-", dir=OUT_DIR.parent))
    try:
        figs, k0, n_keys = {}, {}, {}
        font = wc["tiles"]["font_file"]
        for cand in c["candidates"]:
            blocks, same = [], True
            for row in rows:
                lines = []
                for s in seeds:
                    drawn = {a: [tile(cand, key, s, row) for key in seq(row, a)] for a in authors}
                    for a in authors:
                        lines.append((f"{slug[a]} {s}", drawn[a]))
                    # K0: every tile of a key shared by both rows equals the tile drawn afresh from the key alone
                    for key in sorted(set(seq(row, authors[0])) & set(seq(row, authors[1]))):
                        fresh = tile_fresh(cand, key, s, row)
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
            fh.write(json.dumps({"lexicon_mean": mean, "lexicon_sd": sd, "palette": palette_of, "values": values},
                                indent=1, ensure_ascii=False, allow_nan=False) + "\n")
        res = {"meta": {"conditions_sha256": hashlib.sha256(raw).hexdigest(), "script_sha256": cm.sha256(Path(__file__)),
                        "font_sha256": cm.sha256(Path(font)), **cm.versions("numpy", "scipy", "pillow"), **state, "started": started,
                        "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
               "eps": eps, "K0_same_key_same_tile": k0, "K0_keys_compared": n_keys, "figures": figs,
               "values_sha256": cm.sha256(tmp / "values.json")}
        with (tmp / "sketch.json").open("x") as fh:
            fh.write(json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    except BaseException:
        shutil.rmtree(tmp)
        raise
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} appeared during the run; the output is left in {tmp}")
    tmp.rename(OUT_DIR)
    print(json.dumps(k0))


if __name__ == "__main__":
    main()
