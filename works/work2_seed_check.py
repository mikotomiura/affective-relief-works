"""Work 2, the check of the seeds: the 20 words and the variant tiles of the two trial authors drawn again from their
recorded values under five common seeds and under the seeds of the authors swapped, with sheets without text for the
judgement by eye and the pixel differences that support it.

Usage:  uv run python works/work2_seed_check.py        (refuses to overwrite)
        uv run python works/work2_seed_check.py --figure6   (after the judgement: applies the rule of figure_6 to
        works/conditions/work2_seed_check_answers.json and seed_check.json, writes figure6.json only)
Conditions: works/conditions/work2_seed_check.json, and the field, relief, tiles and figure of
        works/conditions/work2_words.json and works/conditions/work2_variants.json (committed before this script was
        run)
Input:  results/work2_words/words.json and results/work2_variants/variants.json (no text, no lexicon)
Output: results/work2_seed_check/seed_check.json, the labelled figures and the sheets
"""

from __future__ import annotations

import datetime as dt
import hashlib
import itertools
import json
import shutil
import statistics
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.pycache_prefix = tempfile.mkdtemp(prefix="pycache-")  # never run a stale .pyc of src/ (ignored by git)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "works"))
sys.path.insert(0, str(ROOT / "checks"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import _common as cm  # noqa: E402
import relief  # noqa: E402
from atact.mapping import params  # noqa: E402
from work2_variants import figure as variants_figure  # noqa: E402
from work2_variants import row_label  # noqa: E402
from work2_words import tile_figure, z_mm  # noqa: E402

COND = ROOT / "works" / "conditions" / "work2_seed_check.json"
ANSWERS = ROOT / "works" / "conditions" / "work2_seed_check_answers.json"
WORDS_COND = ROOT / "works" / "conditions" / "work2_words.json"
VARIANTS_COND = ROOT / "works" / "conditions" / "work2_variants.json"
OUT_DIR = ROOT / "results" / "work2_seed_check"
CODE = ["works/work2_seed_check.py", "works/work2_words.py", "works/work2_variants.py", "works/_common.py",
        "checks/context_choice.py", "checks/author_texture.py", "src/pipeline.py", "src/japanese_text.py",
        "src/window.py", "src/relief.py", "src/english_text.py", "src/stretch.py"]
WORDS_SET = "words20"


# ---------------------------------------------------------------- differences


def tile_diff(a: np.ndarray, b: np.ndarray, D: float) -> float:
    """Mean absolute difference of two tiles of z (mm), divided by D."""
    if a.shape != b.shape:
        raise ValueError(f"tiles of shapes {a.shape} and {b.shape}")
    return float(np.mean(np.abs(a - b)) / D)


def row_diff(r1: list[np.ndarray], r2: list[np.ndarray], D: float) -> float:
    """Mean over the positions of the differences of two rows of the same length."""
    if len(r1) != len(r2) or not r1:
        raise ValueError(f"rows of lengths {len(r1)} and {len(r2)}")
    return float(np.mean([tile_diff(a, b, D) for a, b in zip(r1, r2)]))


def tiles_differing(t1: dict[str, int], t2: dict[str, int]) -> int:
    """How many tiles two compositions of the same number of tiles differ by."""
    if sum(t1.values()) != sum(t2.values()):
        raise ValueError("compositions of different sizes")
    return sum(t1.values()) - sum(min(t1.get(k, 0), t2.get(k, 0)) for k in set(t1) | set(t2))


def ratios(between_authors: dict[int, float], between_seeds: dict[tuple[str, int, int], float]) -> dict[int, float]:
    """Per common seed S: between_authors(S) / median of between_seeds over both authors and all S' != S."""
    out = {}
    for s, d in between_authors.items():
        ref = [v for (_, s1, s2), v in between_seeds.items() if s in (s1, s2)]
        m = statistics.median(ref)
        if not m > 0:
            raise SystemExit(f"seed {s}: the median difference between seeds is {m}; no ratio")
        out[s] = d / m
    return out


def figure6(sets: list[str], remains: dict[str, bool], judged_invisible: dict[str, bool], median_ba: dict[str, float],
            differing: dict[str, int]) -> dict:
    """The visible and invisible examples by the rule of the conditions (ties by the order of sets)."""

    def best(cands: list[str], key, largest: bool) -> str | None:
        if not cands:
            return None
        return sorted(cands, key=lambda s: ((-1 if largest else 1) * key(s), sets.index(s)))[0]

    vis = [s for s in sets if remains[s]]
    visible = best(vis or sets, median_ba.get, True)
    big = [s for s in sets if differing[s] >= 3]
    inv = [s for s in big if judged_invisible[s]]
    invisible = best(inv or big, median_ba.get, False)
    return {"visible": visible, "visible_remains": bool(vis), "invisible": invisible,
            "invisible_judged": bool(inv), "invisible_eligible": big}


# ---------------------------------------------------------------- sheets


def grey_u8(z: np.ndarray, D: float) -> np.ndarray:
    """The grey of the figures (z / D + 1/2, rows flipped so that the origin is at the lower left)."""
    return np.round(np.flipud(np.clip(z / D + 0.5, 0.0, 1.0)) * 255).astype(np.uint8)


def sheet(lines: list[tuple[list[np.ndarray], list[np.ndarray]]], px: int, D: float, gap: int = 4,
          pair_gap: int = 32, line_gap: int = 16) -> Image.Image:
    """Lines of two rows of tiles (first author, then second), without any text."""
    n = max(len(a) + len(b) for a, b in lines)
    left = max(len(a) for a, _ in lines)
    W = n * px + (n - 2) * gap + pair_gap
    H = len(lines) * px + (len(lines) - 1) * line_gap
    img = Image.new("L", (W, H), 255)
    for i, (a, b) in enumerate(lines):
        y = i * (px + line_gap)
        for j, z in enumerate(a):
            img.paste(Image.fromarray(grey_u8(z, D), mode="L").resize((px, px), Image.BOX), (j * (px + gap), y))
        x0 = left * (px + gap) - gap + pair_gap
        for j, z in enumerate(b):
            img.paste(Image.fromarray(grey_u8(z, D), mode="L").resize((px, px), Image.BOX), (x0 + j * (px + gap), y))
    return img


# ---------------------------------------------------------------- figure 6


def check_answers(a: dict, rows: list[str], common: list[int]) -> None:
    """The answers file: researcher, main and deciding, each covering every row and common seed with booleans, and
    deciding equal to the researcher's answers (the researcher decides where the two differ)."""
    if sorted(a) != ["deciding", "main", "researcher"]:
        raise SystemExit(f"the answers file has the keys {sorted(a)}, not researcher, main and deciding")
    for who in ("researcher", "main", "deciding"):
        if sorted(a[who]) != sorted(rows):
            raise SystemExit(f"{who}: the rows answered differ from {rows}")
        answers_to_rule(a[who], rows, common)
    if a["deciding"] != a["researcher"]:
        raise SystemExit("deciding differs from the researcher's answers")


def answers_to_rule(answers: dict, pairs: list[str], common: list[int]) -> tuple[dict[str, bool], dict[str, bool]]:
    """From the deciding answers (per set and common seed, true = visibly different): whether a set remains (yes under
    every seed) and whether it was judged not visible under every seed."""
    rem, inv = {}, {}
    for s in pairs:
        got = answers[s]
        if sorted(got) != sorted(str(x) for x in common) or any(not isinstance(v, bool) for v in got.values()):
            raise SystemExit(f"{s}: the answers must be true or false for every common seed")
        rem[s] = all(got.values())
        inv[s] = not any(got.values())
    return rem, inv


def run_figure6() -> None:
    out = OUT_DIR / "figure6.json"
    c = json.loads(COND.read_bytes())
    cm.same_as_head([COND, ANSWERS, OUT_DIR / "seed_check.json", Path(__file__)])
    res = json.loads((OUT_DIR / "seed_check.json").read_bytes())
    if res["meta"]["conditions_sha256"] != cm.sha256(COND):
        raise SystemExit("seed_check.json was not computed on these conditions")
    a = json.loads(ANSWERS.read_bytes())
    pairs = c["rows"]["pairs"]
    check_answers(a, [*pairs, WORDS_SET], c["seeds"]["common"])
    rem_all, _ = answers_to_rule(a["deciding"], [*pairs, WORDS_SET], c["seeds"]["common"])
    rem, inv = answers_to_rule(a["deciding"], pairs, c["seeds"]["common"])
    remaining = [r for r in [*pairs, WORDS_SET] if rem_all[r]]
    disagree = {r: [x for x in a["researcher"][r] if a["researcher"][r][x] != a["main"][r][x]]
                for r in [*pairs, WORDS_SET]}
    d = res["differences"]
    got = figure6(pairs, rem, inv, {s: d[s]["median_between_authors_common"] for s in pairs},
                  {s: d[s]["tiles_differing"] for s in pairs})
    meta = {"conditions_sha256": cm.sha256(COND), "script_sha256": cm.sha256(Path(__file__)),
            "answers_sha256": cm.sha256(ANSWERS), "seed_check_sha256": cm.sha256(OUT_DIR / "seed_check.json"),
            "git_commit": cm.git("rev-parse", "HEAD")}
    text = json.dumps({"meta": meta, "rows_remaining": remaining, "two_claims": bool(remaining),
                       "disagreements": {r: x for r, x in disagree.items() if x}, "figure6": {
                           "remains": rem, "judged_invisible": inv, **got}}, indent=1, ensure_ascii=False) + "\n"
    with out.open("x") as fh:
        fh.write(text)
    print(text)


# ---------------------------------------------------------------- run


def main() -> None:
    if sys.argv[1:] == ["--figure6"]:
        run_figure6()
        return
    if sys.argv[1:]:
        raise SystemExit(f"unknown arguments {sys.argv[1:]}")
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} exists; refusing to overwrite")
    raw = COND.read_bytes()
    c = json.loads(raw)
    inp = c["inputs"]
    words_path, variants_path = ROOT / inp["words"], ROOT / inp["variants"]
    for p, h in ((words_path, inp["words_sha256"]), (variants_path, inp["variants_sha256"]),
                 (WORDS_COND, inp["work2_words_conditions_sha256"]),
                 (VARIANTS_COND, inp["work2_variants_conditions_sha256"])):
        if cm.sha256(p) != h:
            raise SystemExit(f"{p.relative_to(ROOT)} differs from the one named in the conditions")
    cm.same_as_head([COND, WORDS_COND, VARIANTS_COND, words_path, variants_path, *(ROOT / p for p in CODE)])
    state = cm.git_state(["works", "src", "checks", "results/work2_words", "results/work2_variants", "pyproject.toml",
                          "uv.lock"])
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    wc = json.loads(WORDS_COND.read_bytes())
    vc = json.loads(VARIANTS_COND.read_bytes())
    words = json.loads(words_path.read_bytes())
    variants = json.loads(variants_path.read_bytes())
    if (words["meta"]["conditions_sha256"] != inp["work2_words_conditions_sha256"]
            or variants["meta"]["conditions_sha256"] != inp["work2_variants_conditions_sha256"]):
        raise SystemExit("a result was not computed on the conditions named")
    cm.check_generator(wc["generator"])
    eps = relief.eps_for(relief.Dimensions(**{k: wc["relief"][k] for k in ("L_mm", "D_mm", "s_max", "base_mm",
                                                                         "e_min_mm")}))
    if eps != words["eps"] or eps != variants["eps"]:
        raise SystemExit("eps differs from the records")
    authors = c["authors"]
    if authors != wc["trial_authors"] or authors != vc["authors_drawn"]:
        raise SystemExit("the authors differ from the trial authors of the earlier runs")
    seeds = c["seeds"]
    if seeds["own"] != {a["author"]: a["seed"] for a in wc["authors"] if a["author"] in authors}:
        raise SystemExit("the own seeds differ from work2_words.json")
    if seeds["swapped"] != {authors[0]: seeds["own"][authors[1]], authors[1]: seeds["own"][authors[0]]}:
        raise SystemExit("the swapped seeds are not the own seeds exchanged")
    for a in authors:  # the recorded values give the recorded parameters
        for r in words["authors"][a]["words"]:
            if [float(x) for x in params(r["V"], r["A"])] != [r["f"], r["s"], r["gamma"]]:
                raise SystemExit(f"{a} {r['word']}: params differ from the record")
    for key, v in variants["values"].items():
        if [float(x) for x in params(v["V"], v["A"])] != [v["f"], v["s"], v["gamma"]]:
            raise SystemExit(f"{key}: params differ from the record")
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    D = wc["relief"]["D_mm"]
    slug_of = {x["author"]: x["slug"] for x in wc["authors"]}

    # tiles ------------------------------------------------------------
    tw, kw = wc["tiles"]["tile_px"], wc["tiles"]["supersample"]
    fg = vc["figure"]
    tv, kv = fg["tile_px"], fg["supersample"]
    cache: dict[tuple, np.ndarray] = {}

    def word_tile(a: str, i: int, seed: int) -> np.ndarray:
        if ("w", a, i, seed) not in cache:
            r = words["authors"][a]["words"][i]
            cache["w", a, i, seed] = z_mm(r["V"], r["A"], seed, tw * kw, wc, eps).reshape(tw, kw, tw, kw).mean(
                axis=(1, 3))
        return cache["w", a, i, seed]

    def spelling_tile(key: str, seed: int) -> np.ndarray:
        if ("v", key, seed) not in cache:
            v = variants["values"][key]
            cache["v", key, seed] = z_mm(v["V"], v["A"], seed, tv * kv, wc, eps).reshape(tv, kv, tv, kv).mean(
                axis=(1, 3))
        return cache["v", key, seed]

    sets = [s for _, s in vc["sets"]["chosen"]]
    rows = variants["rows"]

    def seq(s: str, a: str) -> list[str] | None:
        t = rows[s][a].get("tiles")
        return None if t is None else [k for k, n in t.items() for _ in range(n)]

    pairs = [s for s in sets if all(seq(s, a) is not None for a in authors)]
    if pairs != c["rows"]["pairs"]:
        raise SystemExit(f"the sets drawn for both authors {pairs} differ from rows.pairs of the conditions")
    font = wc["tiles"]["font_file"]

    def words_fig(seed_of: dict[str, int], title) -> Image.Image:
        blocks = [(title(a, seed_of[a]), [word_tile(a, i, seed_of[a]) for i in range(len(words["authors"][a]["words"]))],
                   [r["word"] for r in words["authors"][a]["words"]]) for a in authors]
        return tile_figure(blocks, wc, font)

    def variants_fig(seed_of: dict[str, int]) -> Image.Image:
        bl = [(f"{slug_of[a]} (seed {seed_of[a]})", [(row_label(s, rows[s][a]), seq(s, a)) for s in sets], seed_of[a])
              for a in authors]
        return variants_figure(bl, spelling_tile, {**fg, "n_tiles": vc["tiles_per_set"]["n_tiles"]}, D)

    tmp = Path(tempfile.mkdtemp(prefix=".work2_seed_check-", dir=OUT_DIR.parent))
    try:  # a failure leaves no half-written directory under results/
        # S1: the earlier figures, byte for byte
        own = seeds["own"]
        common0 = wc["common_seed"]
        repro = {
            "results/work2_words/tiles_common_seed.png": words_fig(
                {a: common0 for a in authors}, lambda a, s: f"{slug_of[a]} (common seed {s})"),
            "results/work2_words/tiles.png": words_fig(own, lambda a, s: f"{slug_of[a]} (seed {s})"),
            **{f"results/work2_variants/tiles_seed{s}.png": variants_fig({a: s for a in authors})
               for s in vc["seeds"]["common"]},
            "results/work2_variants/tiles_own_seed.png": variants_fig(own),
        }
        reproduced = {}
        for rel, img in repro.items():
            img.save(tmp / "repro.png")
            got = cm.sha256(tmp / "repro.png")
            (tmp / "repro.png").unlink()
            if got != cm.sha256(ROOT / rel):
                raise SystemExit(f"{rel} is not reproduced byte for byte")
            reproduced[rel] = got

        # labelled figures
        figs = {}
        versions = [(f"seed{s}", {a: s for a in authors}) for s in seeds["common"]]
        versions.append(("swapped", seeds["swapped"]))
        for name, seed_of in versions:
            words_fig(seed_of, lambda a, s: f"{slug_of[a]} (seed {s})").save(tmp / f"words_{name}.png")
            variants_fig(seed_of).save(tmp / f"variants_{name}.png")
            figs[f"words_{name}.png"] = cm.sha256(tmp / f"words_{name}.png")
            figs[f"variants_{name}.png"] = cm.sha256(tmp / f"variants_{name}.png")

        # sheets without text: the five common seeds, then the swapped seeds, then the own seeds
        sheet_versions = [*versions, ("own", own)]
        n_words = len(words["authors"][authors[0]]["words"])
        wl = [[word_tile(a, i, so[a]) for i in range(n_words)] for _, so in sheet_versions for a in authors]
        sheet([(wl[2 * i], wl[2 * i + 1]) for i in range(len(sheet_versions))], 80, D).save(tmp / "words_sheet.png")
        figs["words_sheet.png"] = cm.sha256(tmp / "words_sheet.png")
        for s in pairs:
            lines = [([spelling_tile(k, so[authors[0]]) for k in seq(s, authors[0])],
                      [spelling_tile(k, so[authors[1]]) for k in seq(s, authors[1])]) for _, so in sheet_versions]
            sheet(lines, 120, D).save(tmp / f"variants_sheet_{s}.png")
            figs[f"variants_sheet_{s}.png"] = cm.sha256(tmp / f"variants_sheet_{s}.png")

        # pixel differences
        def row_tiles(s: str, a: str, seed: int) -> list[np.ndarray]:
            if s == WORDS_SET:
                return [word_tile(a, i, seed) for i in range(n_words)]
            return [spelling_tile(k, seed) for k in seq(s, a)]

        diffs = {}
        for s in [*pairs, WORDS_SET]:
            ba = {str(v): row_diff(row_tiles(s, authors[0], so[authors[0]]), row_tiles(s, authors[1], so[authors[1]]),
                                   D) for v, so in [*((x, {a: x for a in authors}) for x in seeds["common"]),
                                                    ("swapped", seeds["swapped"]), ("own", own)]}
            bs = {(a, s1, s2): row_diff(row_tiles(s, a, s1), row_tiles(s, a, s2), D)
                  for a in authors for s1, s2 in itertools.combinations(seeds["common"], 2)}
            rt = ratios({x: ba[str(x)] for x in seeds["common"]}, bs)
            d = {"between_authors": ba,
                 "between_seeds": [{"author": a, "seeds": [s1, s2], "value": v} for (a, s1, s2), v in bs.items()],
                 "ratio": {str(x): v for x, v in rt.items()},
                 "median_between_authors_common": statistics.median(ba[str(x)] for x in seeds["common"])}
            if s != WORDS_SET:
                d["tiles_differing"] = tiles_differing(rows[s][authors[0]]["tiles"], rows[s][authors[1]]["tiles"])
                # S2: the tile of a spelling under a seed does not depend on the author: computed again outside the
                # cache, it equals the cached tile used for both rows
                for x in seeds["common"]:
                    for k1 in set(seq(s, authors[0])) & set(seq(s, authors[1])):
                        v = variants["values"][k1]
                        fresh = z_mm(v["V"], v["A"], x, tv * kv, wc, eps).reshape(tv, kv, tv, kv).mean(axis=(1, 3))
                        if not np.array_equal(fresh, spelling_tile(k1, x)):
                            raise SystemExit(f"{s} {k1}: the tile under seed {x} is not reproduced")
            diffs[s] = d

        res = {"meta": {"conditions_sha256": hashlib.sha256(raw).hexdigest(), "script_sha256": cm.sha256(Path(__file__)),
                        "font_sha256": cm.sha256(Path(font)), **cm.versions("numpy", "scipy", "pillow"), **state,
                        "started": started, "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
               "eps": eps, "reproduced": reproduced, "S2_same_spelling_same_tile": True, "pairs": pairs, "differences": diffs, "figures": figs,
               "sheet_lines": [name for name, _ in sheet_versions]}
        text = json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False) + "\n"  # built before the file is opened
        with (tmp / "seed_check.json").open("x") as fh:
            fh.write(text)
    except BaseException:
        shutil.rmtree(tmp)
        raise
    if OUT_DIR.exists():  # rename(2) would replace an empty directory that appeared meanwhile
        raise SystemExit(f"{OUT_DIR} appeared during the run; the output is left in {tmp}")
    tmp.rename(OUT_DIR)
    for s, d in diffs.items():
        print(s, {k: round(v, 4) for k, v in d["between_authors"].items()}, d.get("tiles_differing"))


if __name__ == "__main__":
    main()
