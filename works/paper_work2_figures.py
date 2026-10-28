"""Two figures of work 2 for the paper: the 20 words under the own seeds above the common seed (two committed images
stacked), and two rows of variant tiles under the common seed 100 chosen by the rule of figure_6 of the check of the
seeds applied to the deciding judge of the blind judgement.

Usage:  uv run python works/paper_work2_figures.py   (refuses to overwrite)
Conditions: works/conditions/paper_work2_figures.json (committed before this script was run)
Input:  the committed images and records named in the conditions (no text, no lexicon)
Output: results/paper_work2_figures/
"""

from __future__ import annotations

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
from work2_seed_check import figure6  # noqa: E402
from work2_words import z_mm  # noqa: E402

COND = ROOT / "works" / "conditions" / "paper_work2_figures.json"
OUT_DIR = ROOT / "results" / "paper_work2_figures"
CODE = ["works/paper_work2_figures.py", "works/work2_seed_check.py", "works/work2_words.py",
        "works/work2_variants.py", "works/_common.py", "checks/context_choice.py", "checks/author_texture.py",
        "src/pipeline.py", "src/japanese_text.py", "src/window.py", "src/relief.py", "src/english_text.py",
        "src/stretch.py"]
DIFFERENT = "違って見える"


def rule_inputs(answers: dict, sets: list[str], seeds: list[int]) -> tuple[dict[str, bool], dict[str, bool]]:
    """From the judge's answers in one version (per set, per seed as a string): whether a set remains (DIFFERENT under
    every seed) and whether it was judged not visible under every seed (no DIFFERENT under any seed: as in figure_6,
    an answer is true for visibly different and false otherwise)."""
    rem, inv = {}, {}
    for s in sets:
        got = answers[s]["answers"]
        if sorted(got) != sorted(str(x) for x in seeds):
            raise SystemExit(f"{s}: the answers do not cover the common seeds {seeds}")
        rem[s] = all(got[str(x)] == DIFFERENT for x in seeds)
        inv[s] = not any(got[str(x)] == DIFFERENT for x in seeds)
    return rem, inv


def tile_box(block: int, row: int, k: int, f: dict) -> tuple[int, int, int, int]:
    """The box of a tile in the figure of works/work2_variants.py (head 36, a row of label, tiles and tile label)."""
    px, gap, n = f["tile_px"], f["gap_px"], f["n_tiles"]
    bw = n * px + (n - 1) * gap
    x = block * (bw + f["block_gap_px"]) + k * (px + gap)
    y = 36 + row * (f["row_label_px"] + px + f["tile_label_px"] + f["row_gap_px"]) + f["row_label_px"]
    return x, y, x + px, y + px


def judged_figure(blocks: list[tuple[str, list[tuple[str, list[np.ndarray], list[str], str]]]], f: dict,
                  font_file: str) -> Image.Image:
    """blocks: (title, rows); a row is (label, tiles as grey u8 arrays, spellings, key line)."""
    px, gap = f["tile_px"], f["gap_px"]
    n = max(len(t) for _, rows in blocks for _, t, _, _ in rows)
    bw = n * px + (n - 1) * gap
    head, rl, tl, kl, rg = 36, 30, 26, 24, 18
    rh = rl + px + tl + kl
    nrows = max(len(rows) for _, rows in blocks)
    W = len(blocks) * bw + (len(blocks) - 1) * f["block_gap_px"]
    H = head + nrows * rh + (nrows - 1) * rg
    img = Image.new("L", (W, H), 255)
    draw = ImageDraw.Draw(img)
    big, small = ImageFont.truetype(font_file, 18), ImageFont.truetype(font_file, 14)
    for b, (title, rows) in enumerate(blocks):
        x0 = b * (bw + f["block_gap_px"])
        draw.text((x0, 6), title, fill=0, font=big)
        for r, (label, tiles, spellings, key) in enumerate(rows):
            y0 = head + r * (rh + rg)
            draw.text((x0, y0 + 6), label, fill=0, font=small)
            for k, (t, sp) in enumerate(zip(tiles, spellings)):
                x, y = x0 + k * (px + gap), y0 + rl
                img.paste(Image.fromarray(t, mode="L"), (x, y))
                draw.text((x, y + px + 3), sp, fill=0, font=small)
            draw.text((x0, y0 + rl + px + tl), key, fill=0, font=small)
    return img


def main() -> None:
    if sys.argv[1:]:
        raise SystemExit(f"unknown arguments {sys.argv[1:]}")
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} exists; refusing to overwrite")
    raw, c = cm.load_conditions(COND)
    inp = c["inputs"]
    path = {k: ROOT / v["path"] for k, v in inp.items() if isinstance(v, dict)}
    for k in ("own_seeds", "common_seed", "variants_image"):
        cm.check_file(path[k], inp[k]["sha256"], k)
    cm.same_as_head([COND, *path.values(), *(ROOT / p for p in CODE)])
    state = cm.git_state(["works", "src", "checks", "results/work2_words", "results/work2_variants", "results/work2_seed_check",
                          "results/blind_judgement", "pyproject.toml", "uv.lock"])
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    sc = json.loads(path["seed_check"].read_bytes())
    scc = json.loads(path["seed_check_conditions"].read_bytes())
    vc = json.loads(path["variants_conditions"].read_bytes())
    wc = json.loads(path["words_conditions"].read_bytes())
    variants = json.loads(path["variants"].read_bytes())
    score = json.loads(path["score"].read_bytes())
    if sc["meta"]["conditions_sha256"] != cm.sha256(path["seed_check_conditions"]):
        raise SystemExit("seed_check.json was not computed on the conditions named")
    if variants["meta"]["conditions_sha256"] != cm.sha256(path["variants_conditions"]):
        raise SystemExit("variants.json was not computed on the conditions named")
    if score["meta"]["conditions_sha256"] != cm.sha256(path["score_conditions"]):
        raise SystemExit("score.json was not computed on the conditions named")
    if vc["work2_words_conditions_sha256"] != cm.sha256(path["words_conditions"]):
        raise SystemExit("work2_variants.json names other conditions of the 20 words")
    for name, r in (("seed_check", sc), ("variants", variants), ("score", score)):
        if r["meta"].get("git_dirty") is not False:
            raise SystemExit(f"{name}: not recorded as computed on a clean tree")
    if sc["pairs"] != scc["rows"]["pairs"] or scc["authors"] != vc["authors_drawn"]:
        raise SystemExit("the sets or the authors differ between the records")
    for k in ("own_seeds", "common_seed", "variants_image"):
        if sc["reproduced"][inp[k]["path"]] != inp[k]["sha256"]:
            raise SystemExit(f"{k}: the SHA-256 differs from the one reproduced by the check of the seeds")
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    # (a) the stack ------------------------------------------------------
    top, bottom = Image.open(path["own_seeds"]), Image.open(path["common_seed"])
    if top.mode != "L" or bottom.mode != "L" or top.width != bottom.width:
        raise SystemExit(f"the two images differ in mode or width: {top.mode} {top.size}, {bottom.mode} {bottom.size}")
    stack = Image.new("L", (top.width, top.height + 24 + bottom.height), 255)
    stack.paste(top, (0, 0))
    stack.paste(bottom, (0, top.height + 24))

    # (b) the rule ------------------------------------------------------
    jr = c["judged_rows"]
    if score["decision"].get("decided") is not True or not score["judges"][jr["judge"]]["controls"]["pass"]:
        raise SystemExit("the blind judgement did not decide")
    sets, seeds = sc["pairs"], scc["seeds"]["common"]
    answers = score["judges"][jr["judge"]]["test"][jr["version"]]
    rem, inv = rule_inputs(answers, sets, seeds)
    d = sc["differences"]
    got = figure6(sets, rem, inv, {s: d[s]["median_between_authors_common"] for s in sets},
                  {s: d[s]["tiles_differing"] for s in sets})
    chosen = [got["visible"], got["invisible"]]
    if chosen != jr["expected"]:
        raise SystemExit(f"the rule gives {chosen}, not {jr['expected']} as read before the run")
    if any(rem[s] != answers[s]["remains"] for s in sets):
        raise SystemExit("the rows that remain differ from score.json")
    f6 = json.loads(path["figure6"].read_bytes())["figure6"]
    if [f6["visible"], f6["invisible"]] != chosen:
        raise SystemExit("the rows differ from figure6.json of the check of the seeds")

    # (b) the tiles -----------------------------------------------------
    cm.check_generator(wc["generator"])
    eps = relief.eps_for(relief.Dimensions(**{k: wc["relief"][k] for k in ("L_mm", "D_mm", "s_max", "base_mm",
                                                                         "e_min_mm")}))
    if eps != variants["eps"]:
        raise SystemExit("eps differs from the record")
    D = wc["relief"]["D_mm"]
    authors = scc["authors"]
    slug_of = {x["author"]: x["slug"] for x in wc["authors"]}
    fg = {**vc["figure"], "n_tiles": vc["tiles_per_set"]["n_tiles"]}
    tv, kv = fg["tile_px"], fg["supersample"]
    order = [s for _, s in vc["sets"]["chosen"]]
    src = np.asarray(Image.open(path["variants_image"]))
    seed = jr["seed"]
    rows_out = {}
    blocks = []
    for b, a in enumerate(authors):
        rows = []
        for s in chosen:
            row = variants["rows"][s][a]
            seq = [k for k, n in row["tiles"].items() for _ in range(n)]
            tiles = []
            for i, key in enumerate(seq):
                x0, y0, x1, y1 = tile_box(b, order.index(s), i, fg)
                cut = src[y0:y1, x0:x1]
                v = variants["values"][key]
                z = z_mm(v["V"], v["A"], seed, tv * kv, wc, eps).reshape(tv, kv, tv, kv).mean(axis=(1, 3))
                fresh = np.round(np.flipud(np.clip(z / D + 0.5, 0.0, 1.0)) * 255).astype(np.uint8)
                if not np.array_equal(cut, fresh):
                    raise SystemExit(f"{a} {s} tile {i} ({key}): the cut differs from the tile computed again")
                tiles.append(cut)
            n, (k, m) = row["n"], row["works"]
            label = f"{jr['row_names'][s]}: n {n} in {k} of {m} works"
            key_line = " / ".join(f"{sp} {jr['romanization'][sp]} x{t} (V {variants['values'][sp]['V']:.2f}, "
                                  f"A {variants['values'][sp]['A']:.2f})" for sp, t in row["tiles"].items() if t > 0)
            rows.append((label, tiles, seq, key_line))
            rows_out[f"{a}/{s}"] = {"label": label, "spellings": seq, "key": key_line}
        blocks.append((f"{slug_of[a]} (common seed {seed})", rows))
    fig = judged_figure(blocks, fg, jr["font_file"])

    tmp = Path(tempfile.mkdtemp(prefix=".paper_work2_figures-", dir=OUT_DIR.parent))
    try:  # a failure leaves no half-written directory under results/
        stack.save(tmp / c["stack"]["file"])
        fig.save(tmp / jr["file"])
        res = {"meta": {"conditions_sha256": hashlib.sha256(raw).hexdigest(), "script_sha256": cm.sha256(Path(__file__)),
                        "inputs_sha256": {k: cm.sha256(p) for k, p in path.items()},
                        "font_sha256": cm.sha256(Path(jr["font_file"])), **cm.versions("numpy", "pillow"), **state,
                        "started": started,
                        "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
               "rule": {"judge": jr["judge"], "version": jr["version"], "remains": rem, "judged_invisible": inv, **got},
               "rows": rows_out,
               "outputs_sha256": {n: cm.sha256(tmp / n) for n in (c["stack"]["file"], jr["file"])}}
        text = json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
        with (tmp / "paper_work2_figures.json").open("x") as fh:
            fh.write(text)
    except BaseException:
        shutil.rmtree(tmp)
        raise
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} appeared during the run; the output is left in {tmp}")
    tmp.rename(OUT_DIR)
    print(json.dumps(res["rule"], ensure_ascii=False))


if __name__ == "__main__":
    main()
