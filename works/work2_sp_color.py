"""Work 2, the main images on the direction chosen (decision 1 of .steering/20261028-color-features): the 20 words of
each of the six authors, the relief SP (V and A as author-equal percentiles among the words of the works, a seed per
word) in grey, the same relief coloured in OKLCH (SPC) and with the hue band reversed (SPC_reversed, a supplement),
and the coloured sheets as seen with simulated colour vision deficiencies.

Usage:  uv run --with colorspacious==1.1.2 python works/work2_sp_color.py        (refuses to overwrite)
Conditions: works/conditions/work2_sp_color.json (committed before this script was run)
Input:  the 23 works of work 2 (as works/work2_words.py), the lexicon of work 2, results/work2_words/words.json
Output: results/work2_sp_color/
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import sys
import tempfile
from importlib import metadata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.pycache_prefix = tempfile.mkdtemp(prefix="pycache-")  # never run a stale .pyc of src/ (ignored by git)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "works"))
sys.path.insert(0, str(ROOT / "checks"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import _common as cm  # noqa: E402
import japanese_text  # noqa: E402
import relief  # noqa: E402
from atact.mapping import params  # noqa: E402
from pipeline import calibration_map, read_aozora_zip, work2_body, work2_text  # noqa: E402
from sketch_color_features import grey_u8, spelling_seed  # noqa: E402
from sketch_sp_color import (linear_to_oklab, oklch_tint, reference_weights, srgb_to_linear,  # noqa: E402
                             weighted_percentile)
from work2_words import z_mm  # noqa: E402

COND = ROOT / "works" / "conditions" / "work2_sp_color.json"
WORDS_COND = ROOT / "works" / "conditions" / "work2_words.json"
WORK2_COND = ROOT / "works" / "conditions" / "work2.json"
WORK2_RESULT = ROOT / "results" / "work2" / "work2.json"
SKETCH_VALUES = ROOT / "results" / "sketch_sp_color" / "values.json"
SKETCH_RECORD = ROOT / "results" / "sketch_sp_color" / "sketch.json"
SKETCH_COND = ROOT / "works" / "conditions" / "sketch_sp_color.json"
OUT_DIR = ROOT / "results" / "work2_sp_color"
CODE = ["works/work2_sp_color.py", "works/sketch_sp_color.py", "works/sketch_color_features.py",
        "works/work2_words.py", "works/_common.py", "src/pipeline.py", "src/japanese_text.py", "src/relief.py"]
VERSIONS = ("SP", "SPC", "SPC_reversed")
CVD = ("protanomaly", "deuteranomaly", "tritanomaly")


def reference_counts(w2: dict, lex, tagger, vmap, select: str) -> dict[str, Counter]:
    """Per author, the counts of the lexicon entries among the tokens of its works (as results/sketch_sp_color)."""
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
        everything = japanese_text.text_path(t.sentences, lex, tagger, vmap, select=select, order="word")
        counts.setdefault(w["author"], Counter()).update(x for p in everything for x in p.matched)
    if len(counts) != 6:
        raise SystemExit(f"expected six authors, got {len(counts)}")
    return counts


def colour_of(p_valence: float, p_arousal: float, col: dict, version: str) -> tuple[float, float]:
    """Hue in degrees and chroma of the conditions for a coloured version."""
    h = col["hue"][version]
    return (h["hue0"] + h["hue_span"] * p_valence, col["chroma0"] + col["chroma_span"] * p_arousal)


def oklch_chroma(rgb: np.ndarray) -> np.ndarray:
    """OKLCH chroma of sRGB uint8 pixels."""
    lab = linear_to_oklab(srgb_to_linear(rgb.astype(float) / 255.0))
    return np.hypot(lab[..., 1], lab[..., 2])


def simulate(rgb: np.ndarray, cvd_type: str, severity: float, share: bool = False):
    """sRGB uint8 as seen with a colour vision deficiency (Machado et al. 2009, as implemented by colorspacious);
    with share, also the share of values outside [0, 1] before clipping."""
    from colorspacious import cspace_convert
    out = cspace_convert(rgb.astype(float) / 255.0, {"name": "sRGB1+CVD", "cvd_type": cvd_type,
                                                     "severity": severity}, "sRGB1")
    u8 = np.round(np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)
    return (u8, float(((out < 0) | (out > 1)).mean())) if share else u8


def band_check(col: dict, b: dict, severity: float) -> dict:
    """Per coloured version, grey level and chroma: the adjacent and end-to-end CAM02-UCS distances of the band at
    b["steps"] equal steps of p_valence, for normal vision and each simulated deficiency (colours made as the tiles)."""
    from colorspacious import cspace_convert
    out: dict = {}
    for version in col["hue"]:
        for g in b["grey_levels"]:
            for c in b["chroma"]:
                p = np.linspace(0.0, 1.0, b["steps"])
                h = col["hue"][version]
                made = [oklch_tint(np.full((1, 1), g, np.uint8), h["hue0"] + h["hue_span"] * x, c) for x in p]
                rgb = np.concatenate([m[0] for m in made], axis=0)  # steps x 1 x 3
                for vision in ("normal", *CVD):
                    seen, clipped = (rgb, 0.0) if vision == "normal" else simulate(rgb, vision, severity, True)
                    lab = cspace_convert(seen[:, 0].astype(float) / 255.0, "sRGB1", "CAM02-UCS")
                    adj = np.linalg.norm(np.diff(lab, axis=0), axis=1)
                    out[f"{version}/grey{g}/C{c}/{vision}"] = {
                        "adjacent": [round(float(x), 2) for x in adj], "min_adjacent": round(float(adj.min()), 2),
                        "ends": round(float(np.linalg.norm(lab[-1] - lab[0])), 2),
                        "chroma_reduced": [m[1] > 0 for m in made], "share_clipped_by_simulation": clipped}
    return out


def tile_sheet(blocks: list[tuple[str, list[np.ndarray], list[str]]], t: dict, font_path: str) -> Image.Image:
    """Blocks of tiles (RGB uint8) with a title and a numbered label under each tile, laid out in block rows."""
    px, gap, lab, cols, rows = t["tile_px"], t["gap_px"], t["label_px"], t["columns"], t["rows"]
    per_row = t["blocks_per_row"]
    bw, bh = cols * px + (cols - 1) * gap, 40 + rows * (px + lab) + (rows - 1) * gap
    nb_rows = -(-len(blocks) // per_row)
    W = per_row * bw + (per_row - 1) * t["block_gap_px"]
    H = nb_rows * bh + (nb_rows - 1) * t["block_gap_px"]
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, 18)
    for b, (title, tiles, labels) in enumerate(blocks):
        br, bc = divmod(b, per_row)
        x0, y0 = bc * (bw + t["block_gap_px"]), br * (bh + t["block_gap_px"])
        draw.text((x0, y0 + 8), title, fill=(0, 0, 0), font=font)
        for k, (rgb, word) in enumerate(zip(tiles, labels)):
            r, col = divmod(k, cols)
            x, y = x0 + col * (px + gap), y0 + 40 + r * (px + lab + gap)
            img.paste(Image.fromarray(rgb, mode="RGB"), (x, y))
            draw.text((x, y + px + 4), f"{k + 1} {word}", fill=(0, 0, 0), font=font)
    return img


def main() -> None:
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} exists; refusing to overwrite")
    raw = COND.read_bytes()
    c = json.loads(raw)
    if tuple(c["versions"]) != VERSIONS:
        raise SystemExit(f"unknown or missing versions: {list(c['versions'])}")
    _, w2 = cm.load_conditions(WORK2_COND)
    words_path = ROOT / c["inputs"]["words"]
    cm.same_as_head([COND, WORK2_COND, WORDS_COND, WORK2_RESULT, words_path, SKETCH_VALUES, SKETCH_RECORD, SKETCH_COND,
                     ROOT / w2["map"]["file"], *(ROOT / p for p in CODE)])
    state = cm.git_state(["works", "src", "checks", "results/work2", "results/work2_words",
                          "results/sketch_sp_color", "results/calibrate_memolon.json", "pyproject.toml", "uv.lock"])
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
    if words["meta"]["conditions_sha256"] != cm.sha256(WORDS_COND):
        raise SystemExit("the words were not computed on the conditions named")
    eps = relief.eps_for(relief.Dimensions(**{k: wc["relief"][k] for k in ("L_mm", "D_mm", "s_max", "base_mm",
                                                                         "e_min_mm")}))
    if eps != words["eps"]:
        raise SystemExit("eps differs from the record")
    sketch = json.loads(SKETCH_RECORD.read_bytes())
    if (sketch["values_sha256"] != cm.sha256(SKETCH_VALUES)
            or sketch["meta"]["conditions_sha256"] != cm.sha256(SKETCH_COND)):
        raise SystemExit("results/sketch_sp_color/values.json is not the one recorded on its conditions")
    D, seed, col, t = wc["relief"]["D_mm"], c["common_seed"], c["colour"], c["tiles"]
    if t["tile_px"] != wc["tiles"]["tile_px"]:
        raise SystemExit("tile_px differs from that of work2_words.json, whose sampling it relies on")
    if tuple(col["hue"]) != VERSIONS[1:]:
        raise SystemExit("the hue bands do not name the coloured versions")
    if metadata.version("colorspacious") != c["cvd"]["colorspacious"]:
        raise SystemExit("another version of colorspacious than the conditions name")
    authors = list(words["authors"])
    if authors != list(dict.fromkeys(w["author"] for w in w2["works"])) or len(authors) != 6:
        raise SystemExit("the authors differ from those of work 2")
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    counts = reference_counts(w2, lex, tagger, vmap, c["reference"]["select"])
    wts = reference_weights(counts)
    ref_words = sorted(wts)
    vals = {k: np.array([lex.entries[w][i] for w in ref_words]) for k, i in (("valence", 0), ("arousal", 1))}
    weight = np.array([wts[w]["author_equal"] for w in ref_words])

    values: dict[str, dict] = {}
    for a in authors:
        if len(words["authors"][a]["words"]) != t["columns"] * t["rows"]:
            raise SystemExit(f"{a}: the words do not fill a block of tiles")
        for w in words["authors"][a]["words"]:
            k = w["word"]
            if [float(x) for x in params(w["V"], w["A"])] != [w["f"], w["s"], w["gamma"]]:
                raise SystemExit(f"{k}: params differ from the record")
            if k not in wts:
                raise SystemExit(f"{k} does not occur in the reference")
            if tuple(float(x) for x in vmap(*lex.entries[k])) != (w["V"], w["A"]):
                raise SystemExit(f"{k}: the entry does not give the recorded V and A")
            rv, ra = lex.entries[k]
            v = {"count": wts[k]["tokens"], "V_recorded_map_B": w["V"], "A_recorded_map_B": w["A"],
                 "p_valence": weighted_percentile(vals["valence"], weight, rv),
                 "p_arousal": weighted_percentile(vals["arousal"], weight, ra), "seed": spelling_seed(k, seed)}
            if k in values and values[k] != v:
                raise SystemExit(f"{k}: two values")
            values[k] = v

    # R1: the percentiles of the words shared with the sketches equal those recorded there
    skv = json.loads(SKETCH_VALUES.read_bytes())
    if (skv["reference_words"] != len(ref_words)
            or skv["reference_tokens"] != {a: sum(n.values()) for a, n in counts.items()}):
        raise SystemExit("the reference differs from that of results/sketch_sp_color")
    sk = {k: v for k, v in skv["values"].items() if v["entry"] == k}
    shared = sorted(set(values) & set(sk))
    for k in shared:
        if (values[k]["p_valence"], values[k]["p_arousal"]) != (sk[k]["p_valence_author_equal"],
                                                                sk[k]["p_arousal_author_equal"]):
            raise SystemExit(f"{k}: the percentiles differ from results/sketch_sp_color")
    if not shared:
        raise SystemExit("no word shared with the sketches; R1 cannot be applied")

    px, ss = t["tile_px"], wc["tiles"]["supersample"]
    reduced: dict[str, list[float]] = {v: [] for v in VERSIONS if v != "SP"}
    realised: dict[str, dict] = {}

    def tile(k: str, version: str, record: bool = True) -> np.ndarray:
        v = values[k]
        z = z_mm(2 * v["p_valence"] - 1, 2 * v["p_arousal"] - 1, v["seed"], px * ss, wc, eps)
        g = grey_u8(z.reshape(px, ss, px, ss).mean(axis=(1, 3)), D)
        if version == "SP":
            return np.repeat(g[..., None], 3, axis=-1)
        hue, chroma = colour_of(v["p_valence"], v["p_arousal"], col, version)
        rgb, share = oklch_tint(g, hue, chroma)
        if record:
            reduced[version].append(share)  # marks the coloured versions; K2b is computed from realised
            realised.setdefault(version, {})[k] = {"requested": chroma, "share_reduced": share,
                                                   "mean_realised": float(oklch_chroma(rgb).mean())}
        return rgb

    tmp = Path(tempfile.mkdtemp(prefix=".work2_sp_color-", dir=OUT_DIR.parent))
    try:
        figs, k0, n_compared = {}, {}, {}
        font = wc["tiles"]["font_file"]
        for version in VERSIONS:
            blocks, drawn = [], {}
            for a in authors:
                ks = [w["word"] for w in words["authors"][a]["words"]]
                for k in ks:
                    if k not in drawn:
                        drawn[k] = tile(k, version)
                tiles = [drawn[k] for k in ks]
                blocks.append((words["authors"][a]["slug"], tiles, ks))
            k0[version] = all(np.array_equal(img, tile(k, version, record=False))  # K0
                              for k, img in drawn.items())
            n_compared[version] = len(drawn)
            name = f"{version}.png"
            sheet = tile_sheet(blocks, t, font)
            sheet.save(tmp / name)
            figs[name] = cm.sha256(tmp / name)
            if version != "SP":  # K5: the sheet as seen with each simulated deficiency
                for vision in CVD:
                    sim = f"{version}_{vision}.png"
                    Image.fromarray(simulate(np.asarray(sheet), vision, c["cvd"]["severity"]), mode="RGB").save(tmp / sim)
                    figs[sim] = cm.sha256(tmp / sim)
        if not all(k0.values()):
            raise SystemExit(f"K0 fails: {k0}")
        res = {"meta": {"conditions_sha256": hashlib.sha256(raw).hexdigest(), "script_sha256": cm.sha256(Path(__file__)),
                        "words_sha256": cm.sha256(words_path), "sketch_values_sha256": cm.sha256(SKETCH_VALUES),
                        "font_sha256": cm.sha256(Path(font)), "lexicon_sha256": lex.sha256,
                        "dicdir_sha256": tagger.dicdir_sha256, "sketch_record_sha256": cm.sha256(SKETCH_RECORD),
                        "colorspacious_source": "uv run --with (outside uv.lock)",
                        **cm.versions("numpy", "scipy", "pillow", "fugashi", "ipadic", "colorspacious"), **state,
                        "started": started, "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
               "eps": eps, "reference_words": len(ref_words),
               "reference_tokens": {a: sum(n.values()) for a, n in counts.items()},
               "R1_words_shared_with_sketches": len(shared), "K0_same_word_same_tile": k0,
               "K0_words_compared": n_compared,
               "tiles": {a: [w["word"] for w in words["authors"][a]["words"]] for a in authors},
               "distinct_words": len(values),
               "shared_between_authors": sorted(k for k in values
                                                if sum(k in [w["word"] for w in words["authors"][a]["words"]]
                                                       for a in authors) > 1),
               "K2b_note": "over the 120 tiles as laid out (a word of two authors counts twice); K2c is per distinct word",
               "K2b_share_of_pixels_with_chroma_reduced": {
                   v: float(np.mean([realised[v][w["word"]]["share_reduced"]
                                     for a in authors for w in words["authors"][a]["words"]])) for v in reduced},
               "values_note": "V_ and A_recorded_map_B are the values of results/work2_words (checked); the relief is drawn from V = 2 p_valence - 1 and A = 2 p_arousal - 1",
               "K2c_chroma_requested_and_realised": realised,
               "K5_band": band_check(col, c["cvd"]["band"], c["cvd"]["severity"]), "values": values, "figures": figs}
        with (tmp / "work2_sp_color.json").open("x") as fh:
            fh.write(json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    except BaseException:
        shutil.rmtree(tmp)
        raise
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} appeared during the run; the output is left in {tmp}")
    tmp.rename(OUT_DIR)
    print(json.dumps({"K0": k0, "K2b": res["K2b_share_of_pixels_with_chroma_reduced"], "R1": len(shared)}))


if __name__ == "__main__":
    main()
