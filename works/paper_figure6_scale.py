"""Figure 6 of the paper on the two works, rebuilt after round 4: the 20 words of the two trial authors in grey under the
common seed 100, map B above (version G0 of the blind judgement) and the author-equal percentiles below (version P).

Usage:  uv run python works/paper_figure6_scale.py   (refuses to overwrite)
Conditions: works/conditions/paper_figure6_scale.json (committed before this script was run)
Input:  the committed values and images named in the conditions (no text, no lexicon)
Output: results/paper_figure6_scale/
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
from PIL import Image  # noqa: E402

import _common as cm  # noqa: E402
import relief  # noqa: E402
from blind_judgement import sketch_tile_place  # noqa: E402
from sketch_color_features import grey_u8  # noqa: E402
from work2_words import tile_figure, z_mm  # noqa: E402

COND = ROOT / "works" / "conditions" / "paper_figure6_scale.json"
OUT_DIR = ROOT / "results" / "paper_figure6_scale"
CODE = ["works/paper_figure6_scale.py", "works/blind_judgement.py", "works/sketch_color_features.py",
        "works/sketch_sp_color.py", "works/work2_words.py", "works/_common.py", "src/relief.py", "src/pipeline.py",
        "src/japanese_text.py", "src/window.py", "src/english_text.py", "src/stretch.py"]
HEAD = 40  # the band of the titles in tile_figure of works/work2_words.py
GAP = 24
VERSIONS = (("G0", "map B"), ("P", "percentiles"))


def va(version: str, v: dict) -> tuple[float, float]:
    """V and A of a word in a version, as the function tile of works/blind_judgement.py."""
    if version == "G0":
        return v["V"], v["A"]
    return 2 * v["p_valence_author_equal"] - 1, 2 * v["p_arousal_author_equal"] - 1


def main() -> None:
    if sys.argv[1:]:
        raise SystemExit(f"unknown arguments {sys.argv[1:]}")
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} exists; refusing to overwrite")
    raw, c = cm.load_conditions(COND)
    inp = c["inputs"]
    path = {k: ROOT / v["path"] for k, v in inp.items() if isinstance(v, dict)}
    for k in ("values", "sketch_G0", "sketch_P", "common_seed_image"):
        cm.check_file(path[k], inp[k]["sha256"], k)
    cm.same_as_head([COND, *path.values(), *(ROOT / p for p in CODE)])
    state = cm.git_state(["works", "src", "checks", "results/sketch_sp_color", "results/work2_words", "pyproject.toml",
                          "uv.lock"])
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    sk = json.loads(path["sketch_record"].read_bytes())
    skc = json.loads(path["sketch_conditions"].read_bytes())
    words = json.loads(path["words"].read_bytes())
    wc = json.loads(path["words_conditions"].read_bytes())
    jc = json.loads(path["judgement_conditions"].read_bytes())
    if sk["meta"]["conditions_sha256"] != cm.sha256(path["sketch_conditions"]) or sk["meta"].get("git_dirty") is not False:
        raise SystemExit("sketch.json was not computed on the conditions named, on a clean tree")
    if words["meta"]["conditions_sha256"] != cm.sha256(path["words_conditions"]) or words["meta"].get("git_dirty") is not False:
        raise SystemExit("words.json was not computed on the conditions named, on a clean tree")
    if sk["values_sha256"] != inp["values"]["sha256"] or jc["inputs"]["values_sha256"] != inp["values"]["sha256"]:
        raise SystemExit("values.json is not the one of the sketches and of the judgement")
    for k, rec in (("sketch_G0", sk["figures"]["sketch_G0.png"]), ("sketch_P", sk["figures"]["sketch_P.png"]),
                   ("common_seed_image", words["figures"]["tiles_common_seed.png"])):
        if rec != inp[k]["sha256"]:
            raise SystemExit(f"{k}: the SHA-256 differs from its record")
    cm.check_generator(wc["generator"])
    eps = relief.eps_for(relief.Dimensions(**{k: wc["relief"][k] for k in ("L_mm", "D_mm", "s_max", "base_mm",
                                                                         "e_min_mm")}))
    if eps != words["eps"] or eps != sk["eps"]:
        raise SystemExit("eps differs from the records")
    D = wc["relief"]["D_mm"]
    dr = c["drawing"]
    seed = dr["seed"]
    authors = wc["trial_authors"]
    if authors != skc["authors"] or authors != jc["authors"] or seed != wc["common_seed"] or seed not in skc["seeds"]:
        raise SystemExit("the authors or the seed differ between the records")
    slug_of = {x["author"]: x["slug"] for x in wc["authors"]}
    values = json.loads(path["values"].read_bytes())["values"]
    seq = {a: [w["word"] for w in words["authors"][a]["words"]] for a in authors}
    n_tiles = wc["tiles"]["columns"] * wc["tiles"]["rows"]
    if any(len(seq[a]) != n_tiles for a in authors):  # tile_figure and zip would cut a shorter list silently
        raise SystemExit(f"an author has not {n_tiles} words: { {a: len(seq[a]) for a in authors} }")
    started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    # F3: the values ------------------------------------------------------
    for a in authors:
        for w in words["authors"][a]["words"]:
            v = values.get(w["word"])
            if v is None or "p_valence_author_equal" not in v or "p_arousal_author_equal" not in v:
                raise SystemExit(f"F3: {w['word']} has no percentiles in values.json")
            if (v["V"], v["A"]) != (w["V"], w["A"]):
                raise SystemExit(f"F3: {w['word']}: V and A differ between values.json and words.json")

    # F2: the drawing at 72 px equals the sketches -----------------------
    spx, kk = skc["drawing"]["tile_px"], skc["drawing"]["supersample"]
    if kk != wc["tiles"]["supersample"]:
        raise SystemExit("the sketches were drawn at another supersample")
    row_index, f2 = skc["rows"].index("words20"), 0
    lines = [(s, a) for s in skc["seeds"] for a in authors]
    for version, _ in VERSIONS:
        img = np.asarray(Image.open(path[f"sketch_{version}"]).convert("RGB"))
        for a in authors:
            line = lines.index((seed, a))
            for k, word in enumerate(seq[a]):
                V, A = va(version, values[word])
                g = grey_u8(z_mm(V, A, seed, spx * kk, wc, eps).reshape(spx, kk, spx, kk).mean(axis=(1, 3)), D)
                y, x = sketch_tile_place(row_index, line, k, spx)
                if not np.array_equal(img[y : y + spx, x : x + spx], np.repeat(g[..., None], 3, axis=-1)):
                    raise SystemExit(f"F2: {version} {a} tile {k} ({word}) differs from the sketch")
                f2 += 1

    if f2 != len(VERSIONS) * len(authors) * n_tiles:
        raise SystemExit(f"F2 compared {f2} tiles, not {len(VERSIONS) * len(authors) * n_tiles}")

    # the two halves -----------------------------------------------------
    tp, k = wc["tiles"]["tile_px"], wc["tiles"]["supersample"]
    halves = {}
    for version, name in VERSIONS:
        blocks = []
        for a in authors:
            tiles = []
            for word in seq[a]:
                V, A = va(version, values[word])
                tiles.append(z_mm(V, A, seed, tp * k, wc, eps).reshape(tp, k, tp, k).mean(axis=(1, 3)))
            blocks.append((f"{slug_of[a]}, {name} (common seed {seed})", tiles, seq[a]))
        halves[version] = tile_figure(blocks, wc, dr["font_file"])

    # F1: the upper half is the committed figure with other titles --------
    ref = np.asarray(Image.open(path["common_seed_image"]))
    top = np.asarray(halves["G0"])
    if top.shape != ref.shape or not np.array_equal(top[HEAD:], ref[HEAD:]):
        raise SystemExit("F1: the upper half differs from tiles_common_seed.png below the titles")
    upper, lower = halves["G0"], halves["P"]
    if upper.size != lower.size or upper.mode != "L" or lower.mode != "L":
        raise SystemExit("the two halves differ in size or mode")
    fig = Image.new("L", (upper.width, upper.height + GAP + lower.height), 255)
    fig.paste(upper, (0, 0))
    fig.paste(lower, (0, upper.height + GAP))

    tmp = Path(tempfile.mkdtemp(prefix=".paper_figure6_scale-", dir=OUT_DIR.parent))
    try:  # a failure leaves no half-written directory under results/
        name = c["outputs"]["figure"]
        fig.save(tmp / name)
        res = {"meta": {"conditions_sha256": hashlib.sha256(raw).hexdigest(), "script_sha256": cm.sha256(Path(__file__)),
                        "inputs_sha256": {k: cm.sha256(p) for k, p in path.items()},
                        "font_sha256": cm.sha256(Path(dr["font_file"])), **cm.versions("numpy", "pillow"), **state,
                        "started": started,
                        "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
               "checks": {"F1_upper_equals_committed_below_titles": True,
                          "F2_tiles_equal_to_sketches": f2,
                          "F3_values_agree": sum(len(seq[a]) for a in authors)},
               "words": seq,
               "outputs_sha256": {name: cm.sha256(tmp / name)}}
        text = json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
        with (tmp / "figure6_scale.json").open("x") as fh:
            fh.write(text)
    except BaseException:
        shutil.rmtree(tmp)
        raise
    if OUT_DIR.exists():
        raise SystemExit(f"{OUT_DIR} appeared during the run; the output is left in {tmp}")
    tmp.rename(OUT_DIR)
    print(json.dumps(res["checks"]))


if __name__ == "__main__":
    main()
