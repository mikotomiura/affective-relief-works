"""Sketch of a series of translators, version 3: for each translation, the 20 dictionary-matched words it uses more than
the other two (the statistic of work 2), one tile per word with the word's own dictionary V and A under the rule,
common scale and seed of work 1. A permutation of the translation labels in blocks of runs gives a reference for
whether the selected set can be told from label noise.

Usage:  uv run python works/sketch_translations_words.py        (refuses to overwrite)
Conditions: works/conditions/sketch_translations_words.json, with the texts, extraction, runs, lexicon, generator and
        relief of works/conditions/sketch_translations.json (both committed before this script is run)
Input:  the three texts and the NRC VAD Lexicon v2.1 (external/, not in the repository)
Output: results/sketch_translations_words/sketch_translations_words.json and words.png (the figure is written before
        the record; if a run fails between them, the directory is removed by hand before the run is repeated)

Counting: the body is matched once as a whole by the greedy longest match of atact.lexicon.match_terms; the same loop is
repeated here to keep the position of each term, its list is checked against match_terms, and each term belongs to the
run of its first token. The z of a word is the function log_odds of checks/author_texture.py for the observed
selection; the permutations use a vectorised copy of the same formula, checked against log_odds on the observed counts.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
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
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import _common as cm  # noqa: E402
import relief  # noqa: E402
import sketch_translations as st  # noqa: E402
from atact.field import procedural_field_separable  # noqa: E402
from atact.lexicon import Lexicon, load_lexicon, match_terms, tokenize  # noqa: E402
from author_texture import log_odds  # noqa: E402

COND = ROOT / "works" / "conditions" / "sketch_translations_words.json"
BASE_COND = ROOT / "works" / "conditions" / "sketch_translations.json"
OUT_DIR = ROOT / "results" / "sketch_translations_words"
CODE = ["works/sketch_translations_words.py", "works/sketch_translations.py", "works/_common.py", "src/relief.py",
        "checks/author_texture.py"]
SENS_BLOCKS = (1, 12)
SENS_ALPHA = (500.0, 2000.0)


def match_positions(tokens: list[str], lex: Lexicon) -> list[tuple[int, str]]:
    """(index of the first token, term) for the greedy longest match of atact.lexicon.match_terms."""
    out, i = [], 0
    while i < len(tokens):
        for n in range(min(lex.max_len, len(tokens) - i), 0, -1):
            cand = " ".join(tokens[i : i + n])
            if cand in lex.entries:
                out.append((i, cand))
                i += n
                break
        else:
            i += 1
    return out


def run_counts(tokens: list[str], lex: Lexicon, n: int) -> list[Counter]:
    """Counts of the matched terms per run; a term belongs to the run of its first token (runs cut at k T // n)."""
    pos = match_positions(tokens, lex)
    if [t for _, t in pos] != match_terms(tokens, lex):
        raise SystemExit("the positions of the matched terms do not reproduce match_terms")
    T = len(tokens)
    starts = [k * T // n for k in range(n + 1)]
    out = [Counter() for _ in range(n)]
    k = 0
    for i, term in pos:
        while i >= starts[k + 1]:
            k += 1
        out[k][term] += 1
    return out


def z_matrix(own: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """z of log_odds (checks/author_texture.py) for every slot (rows of own) against the other slots pooled."""
    a0 = prior.sum()
    y1 = own
    y2 = own.sum(axis=0, keepdims=True) - own
    n1 = y1.sum(axis=1, keepdims=True)
    n2 = y2.sum(axis=1, keepdims=True)
    d = np.log((y1 + prior) / (n1 + a0 - y1 - prior)) - np.log((y2 + prior) / (n2 + a0 - y2 - prior))
    return d / np.sqrt(1 / (y1 + prior) + 1 / (y2 + prior))


def candidates(counts: np.ndarray, present: np.ndarray, z: np.ndarray, sel: dict) -> np.ndarray:
    return (counts >= sel["min_count"]) & (present >= sel["min_runs"]) & (z > 0)


def top_stats(z: np.ndarray, cand: np.ndarray, k: int) -> tuple[float, float, bool]:
    """z20 and m20 of one slot; missing values count as 0. Returns (z20, m20, short)."""
    v = np.sort(z[cand])[::-1][:k]
    short = len(v) < k
    v = np.concatenate([v, np.zeros(k - len(v))])
    return float(v[k - 1]), float(v.sum() / k), short


def slot_matrices(per_run: np.ndarray, perm: np.ndarray, block: int) -> tuple[np.ndarray, np.ndarray]:
    """per_run: (3, n, W) counts. perm: (n_blocks, 3), slot s of block b takes translation perm[b, s].
    Returns counts (3, W) and runs present (3, W) of the slots."""
    n = per_run.shape[1]
    src = np.repeat(perm, block, axis=0)[:n]  # (n, 3)
    runs = per_run[src.T, np.arange(n)[None, :]]  # (3, n, W)
    return runs.sum(axis=1), (runs > 0).sum(axis=1)


def permute(per_run: np.ndarray, prior: np.ndarray, sel: dict, block: int, n_perm: int, seed: int) -> dict:
    n = per_run.shape[1]
    if n % block:
        raise SystemExit(f"{n} runs do not split into blocks of {block}")
    rng = np.random.default_rng(seed)
    k = sel["n_words"]
    z20 = np.empty((n_perm, 3))
    m20 = np.empty((n_perm, 3))
    short = 0
    for p in range(n_perm):
        perm = np.array([rng.permutation(3) for _ in range(n // block)])
        counts, present = slot_matrices(per_run, perm, block)
        z = z_matrix(counts.astype(float), prior)
        for s in range(3):
            a, b, sh = top_stats(z[s], candidates(counts[s], present[s], z[s], sel), k)
            z20[p, s], m20[p, s] = a, b
            short += sh
    return {"z20": z20, "m20": m20, "short": short}


def tile_figure(blocks: list[tuple[str, list[np.ndarray], list[str]]], t: dict, font_path: str) -> Image.Image:
    """As works/work2_words.tile_figure, with grey levels already computed (relief.grey, common scale)."""
    px, gap, lab, cols, rows = t["tile_px"], t["gap_px"], t["label_px"], t["columns"], t["rows"]
    bw = cols * px + (cols - 1) * gap
    head = 40
    W = len(blocks) * bw + (len(blocks) - 1) * t["block_gap_px"]
    H = head + rows * (px + lab) + (rows - 1) * gap
    img = Image.new("L", (W, H), 255)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, t["font_size"])
    for b, (title, tiles, labels) in enumerate(blocks):
        x0 = b * (bw + t["block_gap_px"])
        draw.text((x0, 8), title, fill=0, font=font)
        for k, (g, word) in enumerate(zip(tiles, labels)):
            r, col = divmod(k, cols)
            x, y = x0 + col * (px + gap), head + r * (px + lab + gap)
            img.paste(Image.fromarray(np.round(np.flipud(np.clip(g, 0.0, 1.0)) * 255).astype(np.uint8), mode="L"),
                      (x, y))
            draw.text((x, y + px + 4), f"{k + 1} {word}", fill=0, font=font)
    return img


def validate(c: dict) -> None:
    sel, p, t = cm.need(c, "selection", dict), cm.need(c, "permutation", dict), cm.need(c, "tiles", dict)
    cm.need(sel, "alpha_0", float, "selection.")
    for k in ("min_count", "min_runs", "n_words"):
        if cm.need(sel, k, int, "selection.") < 1:
            raise SystemExit(f"selection.{k} is positive")
    for k in ("block_runs", "n_permutations"):
        if cm.need(p, k, int, "permutation.") < 1:
            raise SystemExit(f"permutation.{k} is positive")
    cm.need(p, "seed", int, "permutation.")
    for k in ("tile_px", "gap_px", "label_px", "columns", "rows", "block_gap_px", "font_size"):
        cm.need(t, k, int, "tiles.")
    if t["columns"] * t["rows"] != sel["n_words"]:
        raise SystemExit("the grid of tiles does not hold n_words")
    cm.need(t, "font_file", str, "tiles.")


def run(c: dict, base: dict, base_sha256: str, root: Path, lex: Lexicon, out_dir: Path) -> dict:
    st.validate(base)
    validate(c)
    if c["base_conditions_sha256"] != base_sha256:
        raise SystemExit("the conditions of the first sketch differ from the ones named here")
    sel, pc, tc = c["selection"], c["permutation"], c["tiles"]
    n, r = base["runs"], base["relief"]
    d = st.dimensions(base)
    names = [t["name"] for t in base["translations"]]
    if len(names) != 3:
        raise SystemExit("the permutation is written for three translations")
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
        toks[t["name"]] = tokens
    per = {name: run_counts(tk, lex, n) for name, tk in toks.items()}
    totals = {name: sum(per[name], Counter()) for name in names}
    vocab = sorted(set().union(*totals.values()))
    col = {w: j for j, w in enumerate(vocab)}
    per_run = np.zeros((3, n, len(vocab)), dtype=np.int64)
    for i, name in enumerate(names):
        for k, cnt in enumerate(per[name]):
            for w, m in cnt.items():
                per_run[i, k, col[w]] = m
    pooled = sum(totals.values(), Counter())
    n_pooled = sum(pooled.values())

    def prior_for(a0: float) -> dict[str, float]:
        return {w: a0 * pooled[w] / n_pooled for w in vocab}

    prior = prior_for(sel["alpha_0"])
    prior_vec = np.array([prior[w] for w in vocab])
    counts, present = per_run.sum(axis=1), (per_run > 0).sum(axis=1)
    zmat = z_matrix(counts.astype(float), prior_vec)

    def select(a0: float) -> dict[str, list[str]]:
        p = prior_for(a0)
        out = {}
        for i, name in enumerate(names):
            rest = sum((totals[b] for b in names if b != name), Counter())
            z = log_odds(totals[name], rest, p)
            cand = [w for w in vocab if counts[i, col[w]] >= sel["min_count"]
                    and present[i, col[w]] >= sel["min_runs"] and z[w] > 0]
            out[name] = sorted(cand, key=lambda w: (-z[w], w))
        return out

    # the vectorised z equals log_odds on the observed counts
    for i, name in enumerate(names):
        rest = sum((totals[b] for b in names if b != name), Counter())
        z = log_odds(totals[name], rest, prior)
        if not np.allclose([z[w] for w in vocab], zmat[i], rtol=1e-12, atol=1e-12):
            raise SystemExit("the vectorised z differs from log_odds")
    chosen = select(sel["alpha_0"])
    k = sel["n_words"]
    for name in names:
        if len(chosen[name]) < k:
            raise SystemExit(f"{name}: {len(chosen[name])} candidates, fewer than {k}")
    # permutations (the reference) and the sensitivity, before any figure
    ref = permute(per_run, prior_vec, sel, pc["block_runs"], pc["n_permutations"], pc["seed"])
    sens_blocks = {b: permute(per_run, prior_vec, sel, b, pc["n_permutations"], pc["seed"] + b) for b in SENS_BLOCKS}
    sens_alpha = {a: select(a) for a in SENS_ALPHA}
    font = tc["font_file"]
    out_dir.mkdir(parents=True)
    rec, blocks = {}, []
    for i, name in enumerate(names):
        words = chosen[name][:k]
        zs = [float(zmat[i, col[w]]) for w in words]
        z20, m20 = zs[-1], sum(zs) / k
        p95 = {"z20": float(np.percentile(ref["z20"][:, i], 95)), "m20": float(np.percentile(ref["m20"][:, i], 95))}
        reading = ("distinguishable from the relabelling of blocks" if z20 > p95["z20"] and m20 > p95["m20"]
                   else "insufficient evidence under this null")
        rest = sum((totals[b] for b in names if b != name), Counter())
        n1, n2 = sum(totals[name].values()), sum(rest.values())
        a0 = sel["alpha_0"]
        rows = []
        for w in words:
            y1, y2, a = totals[name][w], rest[w], prior[w]
            delta = math.log((y1 + a) / (n1 + a0 - y1 - a)) - math.log((y2 + a) / (n2 + a0 - y2 - a))
            V, A = lex.entries[w][0], lex.entries[w][1]
            rows.append({"word": w, "count": y1, "count_others": y2, "runs": int(present[i, col[w]]),
                         "log_odds": delta, "z": float(zmat[i, col[w]]), "V": V, "A": A})
        tiles = [relief.grey(relief.tile(x["V"], x["A"], lambda f: procedural_field_separable(
            f, tc["tile_px"], r["seed"], octaves=relief.OCTAVES, normalize=relief.NORMALIZE), r["eps"]).H, "common", d)
            for x in rows]
        blocks.append((name, tiles, words))
        rec[name] = {
            "matched_tokens": n1, "candidates": len(chosen[name]), "words": rows,
            "sd_V": float(np.std([x["V"] for x in rows])), "sd_A": float(np.std([x["A"] for x in rows])),
            "observed": {"z20": z20, "m20": m20}, "p95": p95, "reading": reading,
            "sensitivity": {
                **{f"block_runs_{b}_p95": {"z20": float(np.percentile(s["z20"][:, i], 95)),
                                           "m20": float(np.percentile(s["m20"][:, i], 95))}
                   for b, s in sens_blocks.items()},
                **{f"alpha_0_{a:g}_overlap": len(set(words) & set(s[name][:k])) for a, s in sens_alpha.items()}}}
    img = tile_figure(blocks, tc, font)
    img.save(out_dir / "words.png")
    shared = Counter(w for name in names for w in chosen[name][:k])
    return {"figure": {"file": "words.png", "sha256": cm.sha256(out_dir / "words.png")},
            "font_sha256": cm.sha256(Path(font)), "vocabulary": len(vocab),
            "slots_short": {"block_runs_6": ref["short"], **{f"block_runs_{b}": s["short"]
                                                              for b, s in sens_blocks.items()}},
            "words_in_more_than_one": sorted(w for w, m in shared.items() if m > 1),
            "n_words_in_more_than_one": sum(1 for m in shared.values() if m > 1), "translations": rec}


def main() -> None:
    out_json = OUT_DIR / "sketch_translations_words.json"
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} exists; refusing to overwrite")
    raw, c = cm.load_conditions(COND)
    cm.need(c, "base_conditions_sha256", str)
    raw_base, base = cm.load_conditions(BASE_COND)
    cm.same_as_head([COND, BASE_COND, *(ROOT / p for p in CODE)])
    state = cm.git_state(["works", "src", "checks", "pyproject.toml", "uv.lock", ".python-version"])
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
    text = json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
    with out_json.open("x") as fh:
        fh.write(text)
    print(json.dumps({name: {k: v for k, v in x.items() if k in ("candidates", "observed", "p95", "reading")}
                      for name, x in res["translations"].items()}, indent=1))


if __name__ == "__main__":
    main()
