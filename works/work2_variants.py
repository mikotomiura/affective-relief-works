"""Work 2, the choice among variants of the same meaning as tiles: for each author and each of five sets of check 6,
ten tiles whose composition follows the share of each spelling; every tile is the relief of the lexicon value of its
spelling under the rule of the 20 words.

Usage:  uv run python works/work2_variants.py        (refuses to overwrite)
Conditions: works/conditions/work2_variants.json, the sets of checks/conditions/check6_context_v2.json, the field and
        relief of works/conditions/work2_words.json and the inputs of works/conditions/work2.json (committed before
        this script was run)
Input:  the 23 Aozora Bunko zips, MEmoLon, the map B and the MeCab dictionary of work 2 (external/, results/), and
        results/check6_context/context_choice_v2.json (the per-work counts of the variants the recount must equal)
Output: results/work2_variants/variants.json, tiles_seed100.png, tiles_seed200.png, tiles_own_seed.png
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import shutil
import sys
import tempfile
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.pycache_prefix = tempfile.mkdtemp(prefix="pycache-")  # never run a stale .pyc of src/ (ignored by git)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "works"))
sys.path.insert(0, str(ROOT / "checks"))

import fugashi  # noqa: E402
import ipadic  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import _common as cm  # noqa: E402
import japanese_text  # noqa: E402
import relief  # noqa: E402
from atact.mapping import params  # noqa: E402
from context_choice import lemmas, match_set  # noqa: E402
from pipeline import calibration_map, read_aozora_zip, work2_body, work2_text  # noqa: E402
from work2_words import z_mm  # noqa: E402

COND = ROOT / "works" / "conditions" / "work2_variants.json"
WORK2_COND = ROOT / "works" / "conditions" / "work2.json"
WORDS_COND = ROOT / "works" / "conditions" / "work2_words.json"
WORK2_RESULT = ROOT / "results" / "work2" / "work2.json"
OUT_DIR = ROOT / "results" / "work2_variants"
CODE = ["works/work2_variants.py", "works/work2_words.py", "works/_common.py", "checks/context_choice.py",
        "checks/author_texture.py", "src/pipeline.py", "src/japanese_text.py", "src/window.py", "src/relief.py", "src/english_text.py", "src/stretch.py"]


# ---------------------------------------------------------------- spellings


def spellings(spec: dict) -> list[tuple[str, str, tuple[str, ...]]]:
    """(variant, key, lemmas) of every spelling of a set, in the order of the conditions."""
    return [(v, "".join(f), tuple(f)) for v, fs in spec["variants"].items() for f in fs]


def by_spelling(spec: dict) -> dict:
    """The spec with every spelling its own variant (same parts of speech, same order)."""
    out = {k: spec[k] for k in ("pos1", "pos2") if k in spec}
    sp = spellings(spec)
    if len({key for _, key, _ in sp}) != len(sp):
        raise ValueError(f"two spellings of a set share a key: {[key for _, key, _ in sp]}")
    out["variants"] = {key: [list(f)] for _, key, f in sp}
    return out


def to_variants(counts: Counter, spec: dict) -> Counter:
    out = Counter()
    for v, key, _ in spellings(spec):
        out[v] += counts[key]
    return out


def entry_of(key: str, lem: tuple[str, ...], entries: dict) -> str | None:
    """The lexicon entry of a spelling: its key, or for two lemmas whose key is not an entry, its first lemma."""
    if key in entries:
        return key
    if len(lem) > 1 and lem[0] in entries:
        return lem[0]
    return None


# ---------------------------------------------------------------- tiles


def allocate(counts: dict[str, int | Fraction], order: list[str], n_tiles: int) -> dict[str, int]:
    """Largest remainder: floor of share x n_tiles, the rest to the largest remainders, ties by order. Exact
    arithmetic (integers or fractions): with floats, equal remainders differ by an ulp and the ties are lost."""
    if any(isinstance(v, float) for v in counts.values()):
        raise TypeError("allocate takes integers or fractions, not floats")
    total = sum(counts.values())
    q = {k: Fraction(counts.get(k, 0)) * n_tiles / total for k in order}
    out = {k: math.floor(q[k]) for k in order}
    rest = n_tiles - sum(out.values())
    for k in sorted(order, key=lambda k: (-(q[k] - out[k]), order.index(k)))[:rest]:
        out[k] += 1
    return out


def row_of(pooled: Counter, equal: dict[str, Fraction], order: list[str], n_tiles: int, n_works: tuple[int, int]) -> dict:
    n = sum(pooled.values())
    row = {"n": n, "works": list(n_works), "counts": {k: pooled[k] for k in order if pooled[k]},
           "share": {k: pooled[k] / n for k in order if pooled[k]} if n else {}}
    if n < n_tiles:
        row.update({"tiles": None, "too_few": True})
        return row
    tiles = allocate(dict(pooled), order, n_tiles)
    eq = allocate(equal, order, n_tiles)
    row.update({"tiles": {k: t for k, t in tiles.items() if t}, "zero_tiles": [k for k in order if pooled[k] and
                                                                               not tiles[k]],
                "equal_weight_share": {k: float(equal[k]) for k in order if equal.get(k)},
                "equal_weight_tiles": {k: t for k, t in eq.items() if t}, "equal_weight_differs": eq != tiles})
    return row


def figure(blocks: list[tuple[str, list[tuple[str, list[str] | None]], int]], tile_of, f: dict,
           D: float) -> Image.Image:
    """blocks: (title, rows, seed); a row is (label, spellings of its tiles, or None for a row with too few);
    tile_of(spelling, seed) is the height in mm."""
    px, gap, tl, rl, rg = f["tile_px"], f["gap_px"], f["tile_label_px"], f["row_label_px"], f["row_gap_px"]
    n = f["n_tiles"]
    bw = n * px + (n - 1) * gap
    head = 36
    rh = rl + px + tl
    nrows = max(len(rows) for _, rows, _ in blocks)
    W = len(blocks) * bw + (len(blocks) - 1) * f["block_gap_px"]
    H = head + nrows * rh + (nrows - 1) * rg
    img = Image.new("L", (W, H), 255)
    draw = ImageDraw.Draw(img)
    big, small = ImageFont.truetype(f["font_file"], 18), ImageFont.truetype(f["font_file"], 14)
    for b, (title, rows, seed) in enumerate(blocks):
        x0 = b * (bw + f["block_gap_px"])
        draw.text((x0, 6), title, fill=0, font=big)
        for r, (label, tiles) in enumerate(rows):
            y0 = head + r * (rh + rg)
            draw.text((x0, y0 + 6), label, fill=0, font=small)
            if tiles is None:
                draw.rectangle((x0, y0 + rl, x0 + bw - 1, y0 + rl + px - 1), outline=160)
                continue
            for k, key in enumerate(tiles):
                x, y = x0 + k * (px + gap), y0 + rl
                grey = np.clip(tile_of(key, seed) / D + 0.5, 0.0, 1.0)
                img.paste(Image.fromarray(np.round(np.flipud(grey) * 255).astype(np.uint8), mode="L"), (x, y))
                draw.text((x, y + px + 3), key, fill=0, font=small)
    return img


def row_label(set_id: str, row: dict) -> str:
    n, (k, m) = row["n"], row["works"]
    if row.get("too_few"):
        return f"{set_id}  n {n} < 10 (not drawn), works {k}/{m}"
    shares = " ".join(f"{key} {s:.2f}" for key, s in row["share"].items())
    zero = f"  0 tiles: {' '.join(row['zero_tiles'])}" if row["zero_tiles"] else ""
    return f"{set_id}  n {n}, works {k}/{m}  {shares}{zero}"


# ---------------------------------------------------------------- run


def main() -> None:
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} exists; refusing to overwrite")
    raw = COND.read_bytes()
    c = json.loads(raw)
    _, w2 = cm.load_conditions(WORK2_COND)
    c6_path, c6r_path = ROOT / c["check6_conditions"], ROOT / c["check6_result"]
    for p, h in ((WORK2_COND, c["work2_conditions_sha256"]), (c6_path, c["check6_conditions_sha256"]),
                 (c6r_path, c["check6_result_sha256"]), (WORDS_COND, c["work2_words_conditions_sha256"])):
        if cm.sha256(p) != h:
            raise SystemExit(f"{p.relative_to(ROOT)} differs from the one named in the conditions")
    cm.same_as_head([COND, WORK2_COND, WORDS_COND, c6_path, c6r_path, WORK2_RESULT, ROOT / w2["map"]["file"],
                     *(ROOT / p for p in CODE)])
    state = cm.git_state(["works", "src", "checks", "results/work2", "results/check6_context",
                          "results/calibrate_memolon.json", "pyproject.toml", "uv.lock"])
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    wc = json.loads(WORDS_COND.read_bytes())
    c6 = json.loads(c6_path.read_bytes())
    ref = json.loads(c6r_path.read_bytes())
    if ref["meta"]["conditions_sha256"] != c["check6_conditions_sha256"]:
        raise SystemExit("the check 6 result was not computed on the check 6 conditions named")
    if c["seeds"]["main"] != c["seeds"]["common"][0]:
        raise SystemExit("the main seed must be the first common seed")
    cm.check_generator(wc["generator"])
    cm.check_versions(w2["tagger"], ("fugashi", "ipadic"))
    tagger = japanese_text.Tagger()
    if tagger.dicdir_sha256 != w2["tagger"]["dicdir_sha256"]:
        raise SystemExit("the files of the MeCab dictionary differ from those of work 2")
    lemma_tagger = fugashi.GenericTagger(ipadic.MECAB_ARGS)  # the same dictionary, for feature 6 (as check 6)
    lex = japanese_text.load_memolon(ROOT / w2["lexicon"]["file"])
    if lex.sha256 != w2["lexicon"]["sha256"]:
        raise SystemExit("the lexicon differs from that of work 2")
    vmap = calibration_map(cm.check_file(ROOT / w2["map"]["file"], w2["map"]["sha256"], "the calibration"))
    done = {w["work_id"]: w for ws in json.loads(WORK2_RESULT.read_text())["authors"].values() for w in ws}
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    A = c6["partA_variants"]
    chosen = [(k, s, A["sets"][k][s]) for k, s in c["sets"]["chosen"]]
    named = {s for k in A["sets"].values() for s in k}
    if (set(c["sets"]["not_chosen"]) | {s for _, s, _ in chosen} != named
            or set(c["sets"]["not_chosen"]) & {s for _, s, _ in chosen} or len(chosen) != len({s for _, s, _ in chosen})):
        raise SystemExit("chosen and not_chosen do not cover the sets of check 6 exactly")
    ref_works = {w["work_id"]: w for w in ref["works"]}

    works = []
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
        r = done[w["work_id"]]
        if len(t.points) != r["sentences"] or sum(p.Z for p in t.points) != r["matched_tokens"]:
            raise SystemExit(f"{w['work_id']}: the text step gives another result than results/work2")
        counts = {s: Counter() for _, s, _ in chosen}
        specs = {s: by_spelling(spec) for _, s, spec in chosen}
        for sent in t.sentences:
            toks = lemmas(lemma_tagger, sent)
            for _, s, _ in chosen:
                counts[s] += match_set(toks, specs[s])
        for k, s, spec in chosen:  # E5: the recount equals check 6 version 2
            if to_variants(counts[s], spec) != Counter(ref_works[w["work_id"]]["variants"][k][s]):
                raise SystemExit(f"{w['work_id']} {s}: the recount by spelling differs from check 6")
        works.append({"author": w["author"], "work_id": w["work_id"], "title": w["title"],
                      "counts": {s: dict(cnt) for s, cnt in counts.items()}})

    authors = sorted({w["author"] for w in works}, key=[w["author"] for w in works].index)
    tp = c["tiles_per_set"]
    min_total = A["min_total"]
    values, rows = {}, {}
    for _, s, spec in chosen:
        order = [key for _, key, _ in spellings(spec)]
        rows[s] = {}
        for a in authors:
            mine = [w for w in works if w["author"] == a]
            inc = [w for w in mine if sum(w["counts"][s].values()) >= min_total]
            pooled = sum((Counter(w["counts"][s]) for w in inc), Counter())
            equal = {key: sum((Fraction(w["counts"][s].get(key, 0), sum(w["counts"][s].values())) for w in inc),
                              Fraction(0)) / len(inc) for key in order} if inc else {}
            row = row_of(pooled, equal, order, tp["n_tiles"], (len(inc), len(mine)))
            ps = ref["partA"][[k for k, x, _ in chosen if x == s][0]]["sets"][s]["pooled_share"][a]
            if ps["n"] != row["n"]:
                raise SystemExit(f"{s} {a}: n differs from the pooled share of check 6")
            rows[s][a] = row
        for v, key, lem in spellings(spec):
            if any(w["counts"][s].get(key) for w in works):
                e = entry_of(key, lem, lex.entries)
                if e is None:
                    raise SystemExit(f"{s}: the spelling {key} occurs and has no entry in the lexicon")
                V, A_ = vmap(*lex.entries[e])
                f, sp, g = (float(x) for x in params(V, A_))
                if key in values:
                    raise SystemExit(f"the spelling {key} occurs in two sets")
                values[key] = {"set": s, "variant": v, "entry": e, "V": V, "A": A_, "f": f, "s": sp, "gamma": g}

    # figures of the authors drawn, written into a temporary directory that becomes OUT_DIR only at the end
    eps = relief.eps_for(relief.Dimensions(**{k: wc["relief"][k] for k in ("L_mm", "D_mm", "s_max", "base_mm",
                                                                         "e_min_mm")}))
    fg = c["figure"]
    k, px = fg["supersample"], fg["tile_px"]
    cache: dict[tuple[str, int], np.ndarray] = {}

    def tile(key: str, seed: int) -> np.ndarray:
        if (key, seed) not in cache:
            v = values[key]
            cache[key, seed] = z_mm(v["V"], v["A"], seed, px * k, wc, eps).reshape(px, k, px, k).mean(axis=(1, 3))
        return cache[key, seed]

    seed_of = {x["author"]: x["seed"] for x in wc["authors"]}
    slug_of = {x["author"]: x["slug"] for x in wc["authors"]}
    drawn = c["authors_drawn"]

    def rows_of(a: str) -> list[tuple[str, list[str] | None]]:
        out = []
        for _, s, spec in chosen:
            row = rows[s][a]
            seq = None if row.get("too_few") else [key for _, key, _ in spellings(spec)
                                                  for _ in range(row["tiles"].get(key, 0))]
            out.append((row_label(s, row), seq))
        return out

    tmp = Path(tempfile.mkdtemp(prefix=".work2_variants-", dir=OUT_DIR.parent))
    figs = {}
    D = wc["relief"]["D_mm"]
    fg_n = {**fg, "n_tiles": tp["n_tiles"]}
    versions_ = [(f"tiles_seed{s}.png", {a: s for a in drawn}) for s in c["seeds"]["common"]]
    versions_.append(("tiles_own_seed.png", {a: seed_of[a] for a in drawn}))
    try:  # a failure leaves no half-written directory under results/
        for name, seeds in versions_:
            bl = [(f"{slug_of[a]} (seed {seeds[a]})", rows_of(a), seeds[a]) for a in drawn]
            figure(bl, tile, fg_n, D).save(tmp / name)
            figs[name] = cm.sha256(tmp / name)

        res = {"meta": {"conditions_sha256": hashlib.sha256(raw).hexdigest(), "script_sha256": cm.sha256(Path(__file__)),
                        "work2_conditions_sha256": cm.sha256(WORK2_COND), "lexicon_sha256": lex.sha256,
                        "dicdir_sha256": tagger.dicdir_sha256, "font_sha256": cm.sha256(Path(fg["font_file"])),
                        **cm.versions("fugashi", "ipadic", "numpy", "scipy", "pillow"), **state, "started": started,
                        "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
               "eps": eps, "recount_equals_check6": True, "rows": rows, "values": values,
               "works": works, "figures": figs}
        text = json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False) + "\n"  # built before the file is opened
        with (tmp / "variants.json").open("x") as fh:
            fh.write(text)
    except BaseException:
        shutil.rmtree(tmp)
        raise
    if OUT_DIR.exists():  # rename(2) would replace an empty directory that appeared meanwhile
        raise SystemExit(f"{OUT_DIR} appeared during the run; the output is left in {tmp}")
    tmp.rename(OUT_DIR)
    for _, s, _ in chosen:
        for a in drawn:
            print(s, a, rows[s][a].get("tiles"), rows[s][a]["n"])


if __name__ == "__main__":
    main()
