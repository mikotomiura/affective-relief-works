"""Sketch of a third input on the words of version 3 of the series of translators: the same 20 words per translation,
each drawn with (V, A) as in version 3 and with the dominance D added to gamma (gamma + 0.3 D, the hypothetical third
input of Supplement S6 of the paper on the rule).

Usage:  uv run python works/sketch_translations_words_d.py        (refuses to overwrite)
Conditions: works/conditions/sketch_translations_words_d.json (committed before this script is run)
Input:  the record of version 3 (results/sketch_translations_words/) and the NRC VAD Lexicon v2.1 (external/)
Output: results/sketch_translations_words_d/sketch_translations_words_d.json, words_vad.png and pairs.png (the figures
        are written before the record; if a run fails between them, the directory is removed by hand before the run is
        repeated)

The generator's lexicon keeps only V and A, so D is read here from the fourth column with the same normalisation and
skipping as atact.lexicon.load_lexicon; the V and A read alongside are checked against the generator's entries. The VA
tile is computed by the same formula as relief.tile and checked against it.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.pycache_prefix = tempfile.mkdtemp(prefix="pycache-")  # never run a stale .pyc of src/ (ignored by git)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "works"))
sys.path.insert(0, str(ROOT / "checks"))

import numpy as np  # noqa: E402

import _common as cm  # noqa: E402
import relief  # noqa: E402
import sketch_translations as st  # noqa: E402
import sketch_translations_words as sw  # noqa: E402
from atact.field import procedural_field_separable  # noqa: E402
from atact.lexicon import Lexicon, load_lexicon, tokenize  # noqa: E402
from atact.mapping import params  # noqa: E402
from atact.transform import t_gamma  # noqa: E402

COND = ROOT / "works" / "conditions" / "sketch_translations_words_d.json"
V3_COND = ROOT / "works" / "conditions" / "sketch_translations_words.json"
BASE_COND = ROOT / "works" / "conditions" / "sketch_translations.json"
V3_RECORD = ROOT / "results" / "sketch_translations_words" / "sketch_translations_words.json"
OUT_DIR = ROOT / "results" / "sketch_translations_words_d"
CODE = ["works/sketch_translations_words_d.py", "works/sketch_translations_words.py", "works/sketch_translations.py",
        "works/_common.py", "src/relief.py", "checks/author_texture.py"]


def read_vad(raw: bytes) -> dict[str, tuple[float, float, float]]:
    """term -> (V, A, D), normalised and skipped as atact.lexicon.load_lexicon does; D outside [-1, 1] stops the run."""
    lines = raw.decode("utf-8").splitlines()
    if lines[0].split("\t")[:4] != ["term", "valence", "arousal", "dominance"]:
        raise SystemExit(f"unexpected header {lines[0]!r}")
    out = {}
    for ln in lines[1:]:
        if not ln.strip():
            continue
        term, v, a, d = ln.split("\t")[:4]
        key = " ".join(tokenize(term))
        if key != term.lower().replace("’", "'"):
            continue
        if not -1.0 <= float(d) <= 1.0:
            raise SystemExit(f"dominance outside [-1, 1] for {term!r}")
        out[key] = (float(v), float(a), float(d))
    return out


def heights(V: float, A: float, D: float | None, w: float, F: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """H = s T_{gamma'}(F) (no eps: relief.tile's H, as version 3 drew) with gamma' = gamma + w D (D None: gamma' =
    gamma). Returns H, s, gamma, gamma'."""
    _, s, g = (float(x) for x in params(V, A))
    g2 = g if D is None else g + w * D
    return s * t_gamma(F, g2), s, g, g2


def rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a * a)))


def run(c: dict, v3: dict, base: dict, rec3: dict, lex: Lexicon, vad: dict, out_dir: Path) -> dict:
    st.validate(base)
    sw.validate(v3)
    weight = cm.need(c["third_input"], "gamma_weight", float, "third_input.")
    r, tc = base["relief"], v3["tiles"]
    d = st.dimensions(base)
    names = [t["name"] for t in base["translations"]]
    px = tc["tile_px"]
    rows, blocks, pair_blocks = {}, [], []
    for name in names:
        words = [w["word"] for w in rec3["translations"][name]["words"]]
        out = []
        tiles_vad, pair_tiles, pair_labels = [], [], []
        z_va, z_vad = [], []
        for w in words:
            if w not in lex.entries:
                raise SystemExit(f"{name}: {w!r} is not in the lexicon of the generator")
            V, A = lex.entries[w]
            if w not in vad or (vad[w][0], vad[w][1]) != (V, A):
                raise SystemExit(f"{name}: {w!r} has no D, or its V and A differ from the generator")
            D = vad[w][2]
            F = None

            def field(f: float) -> np.ndarray:
                nonlocal F
                F = procedural_field_separable(f, px, r["seed"], octaves=relief.OCTAVES, normalize=relief.NORMALIZE)
                return F

            H_ref = relief.tile(V, A, field, r["eps"]).H
            H_va, s, g, _ = heights(V, A, None, weight, F)
            if not np.array_equal(H_va, H_ref):
                raise SystemExit("the VA tile differs from relief.tile")
            H_vad, _, _, g2 = heights(V, A, D, weight, F)
            za, zd = relief.scaled(H_va, "common", d), relief.scaled(H_vad, "common", d)
            z_va.append(za)
            z_vad.append(zd)
            out.append({"word": w, "V": V, "A": A, "D": D, "s": s, "gamma": g, "gamma_d": g2,
                        "dH_mm": rms(zd - za), "dH_over_s_mm": rms(zd - za) / s})
            tiles_vad.append(relief.grey(H_vad, "common", d))
            pair_tiles += [relief.grey(H_va, "common", d), relief.grey(H_vad, "common", d)]
            pair_labels += [f"{w} VA", f"{w} VAD"]
        pairs = [(i, j) for i in range(len(z_va)) for j in range(i + 1, len(z_va))]
        between = [rms(z_va[i] - z_va[j]) for i, j in pairs]
        between_vad = [rms(z_vad[i] - z_vad[j]) for i, j in pairs]
        Ds = [x["D"] for x in out]
        gs, g2s = [x["gamma"] for x in out], [x["gamma_d"] for x in out]
        rows[name] = {"words": out, "median_dH_mm": float(np.median([x["dH_mm"] for x in out])),
                      "median_between_words_mm": float(np.median(between)),
                      "median_between_words_vad_mm": float(np.median(between_vad)),
                      "gamma_sd": float(np.std(gs)), "gamma_d_sd": float(np.std(g2s)),
                      "gamma_spearman": st.spearman(gs, g2s),
                      "pairs_same_va_tile": sum(1 for i, j in pairs if np.array_equal(z_va[i], z_va[j])),
                      "gamma_d_range": [min(g2s), max(g2s)],
                      "D_mean": float(np.mean(Ds)), "D_sd": float(np.std(Ds)), "D_zero": sum(1 for x in Ds if x == 0.0)}
        blocks.append((name, tiles_vad, [f"{w} D{x['D']:+.2f}" for w, x in zip(words, out)]))
        pair_blocks.append((name, pair_tiles, pair_labels))
    out_dir.mkdir(parents=True)
    sw.tile_figure(blocks, tc, tc["font_file"]).save(out_dir / "words_vad.png")
    ptc = dict(tc, columns=8, rows=5)  # 20 pairs, four pairs to a row
    sw.tile_figure(pair_blocks, ptc, tc["font_file"]).save(out_dir / "pairs.png")
    return {"figures": {k: cm.sha256(out_dir / k) for k in ("words_vad.png", "pairs.png")},
            "font_sha256": cm.sha256(Path(tc["font_file"])), "translations": rows}


def main() -> None:
    out_json = OUT_DIR / "sketch_translations_words_d.json"
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} exists; refusing to overwrite")
    raw, c = cm.load_conditions(COND)
    raw3, v3 = cm.load_conditions(V3_COND)
    raw_base, base = cm.load_conditions(BASE_COND)
    if hashlib.sha256(raw3).hexdigest() != cm.need(c, "base_conditions_sha256", str):
        raise SystemExit("the conditions of version 3 differ from the ones named here")
    if hashlib.sha256(raw_base).hexdigest() != cm.need(v3, "base_conditions_sha256", str):
        raise SystemExit("the base conditions differ from the ones version 3 named")
    if cm.sha256(V3_RECORD) != cm.need(c, "words_record_sha256", str):
        raise SystemExit("the record of version 3 differs from the one named here")
    cm.same_as_head([COND, V3_COND, BASE_COND, V3_RECORD, *(ROOT / p for p in CODE)])
    state = cm.git_state(["works", "src", "checks", "results/sketch_translations_words", "pyproject.toml", "uv.lock",
                          ".python-version"])
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    cm.check_generator(base["generator"])
    lex_path = ROOT / base["lexicon"]["file"]
    lex = load_lexicon(lex_path)
    lex_raw = lex_path.read_bytes()
    if lex.sha256 != base["lexicon"]["sha256"] or hashlib.sha256(lex_raw).hexdigest() != lex.sha256:
        raise SystemExit("the lexicon does not match the SHA-256 of the conditions")
    rec3 = json.loads(V3_RECORD.read_text())
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    res = run(c, v3, base, rec3, lex, read_vad(lex_raw), OUT_DIR)
    res = {"meta": {"conditions_sha256": hashlib.sha256(raw).hexdigest(), "lexicon_sha256": lex.sha256,
                    "atact_library_sha256": cm.library_sha256(), "uv_lock_sha256": cm.sha256(ROOT / "uv.lock"),
                    **cm.versions("numpy", "pillow"), **state, "started": started,
                    "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}, **res}
    text = json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
    with out_json.open("x") as fh:
        fh.write(text)
    print(json.dumps({n: {k: v for k, v in x.items() if k != "words"} for n, x in res["translations"].items()},
                     indent=1))


if __name__ == "__main__":
    main()
