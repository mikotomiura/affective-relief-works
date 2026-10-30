"""The figure words_vad.png of the sketch of a third input, drawn again with the words only in its labels (the original
labels also print the dominance D of the NRC VAD Lexicon v2.1 to two decimals, which its terms of use do not let us
redistribute). Conditions: tools/conditions/words_vad_unlabelled.json, committed before this tool.

Usage:  uv run python tools/words_vad_unlabelled.py      (needs the lexicon in external/; refuses to overwrite)
Output: results/sketch_translations_words_d/words_vad_unlabelled.png and words_vad_unlabelled.json

The run of works/sketch_translations_words_d.py is repeated twice in temporary directories: first as it is, and both
figures must have the SHA-256 recorded in results/sketch_translations_words_d/sketch_translations_words_d.json; then
with ' D+0.00' cut from the end of each label of words_vad.png, nothing else changed.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.pycache_prefix = tempfile.mkdtemp(prefix="pycache-")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "works"))
sys.path.insert(0, str(ROOT / "checks"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import _common as cm  # noqa: E402
import sketch_translations_words as sw  # noqa: E402
import sketch_translations_words_d as wd  # noqa: E402
from atact.lexicon import load_lexicon  # noqa: E402

COND = ROOT / "tools" / "conditions" / "words_vad_unlabelled.json"
RECORD = ROOT / "results" / "sketch_translations_words_d" / "sketch_translations_words_d.json"
OUT_PNG = ROOT / "results" / "sketch_translations_words_d" / "words_vad_unlabelled.png"
OUT_JSON = OUT_PNG.with_suffix(".json")
LABEL_D = re.compile(r" D[+-]\d\.\d\d$")


def run_once(inputs: tuple, unlabel: bool) -> dict[str, bytes]:
    """One run of wd.run in a temporary directory; with unlabel, the D is cut from the labels of words_vad.png."""
    original = sw.tile_figure
    cut = []

    def tile_figure(blocks, tc, font):
        if unlabel and not cut:  # the first call draws words_vad.png, the second pairs.png
            new = []
            for name, tiles, labels in blocks:
                if not all(LABEL_D.search(x) for x in labels):
                    raise SystemExit(f"{name}: a label of words_vad.png has no D to cut")
                new.append((name, tiles, [LABEL_D.sub("", x) for x in labels]))
                cut.extend(labels)
            blocks = new
        return original(blocks, tc, font)

    sw.tile_figure = tile_figure
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            wd.run(*inputs, out)
            files = {k: (out / k).read_bytes() for k in ("words_vad.png", "pairs.png")}
    finally:
        sw.tile_figure = original
    if unlabel and len(cut) != 60:
        raise SystemExit(f"{len(cut)} labels cut, not 60")
    return files


def only_labels_removed(old: bytes, new: bytes, tile_rows: int) -> int:
    """Compared in every channel: each changed pixel is white in the new figure (a label's D taken away), and the
    changed rows form at most one band per row of tiles, none taller than 24 px; returns the number of pixels."""
    import io

    a = np.asarray(Image.open(io.BytesIO(old)).convert("RGBA"))
    b = np.asarray(Image.open(io.BytesIO(new)).convert("RGBA"))
    if a.shape != b.shape:
        raise SystemExit("the size of words_vad.png changed")
    d = (a != b).any(axis=2)
    if not d.any() or (b[d][:, :3] != 255).any():
        raise SystemExit("words_vad.png changed in more than the labels")
    rows = np.flatnonzero(d.any(axis=1))
    bands = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
    if len(bands) > tile_rows or any(x[-1] - x[0] + 1 > 24 for x in bands):
        raise SystemExit("the changed pixels are not confined to the bands of the labels")
    return int(d.sum())


def main() -> None:
    if OUT_PNG.exists() or OUT_JSON.exists():
        raise SystemExit(f"{OUT_PNG} exists; refusing to overwrite")
    raw_c, _ = cm.load_conditions(COND)
    _, cd = cm.load_conditions(wd.COND)
    _, v3 = cm.load_conditions(wd.V3_COND)
    _, base = cm.load_conditions(wd.BASE_COND)
    cm.same_as_head([COND, wd.COND, wd.V3_COND, wd.BASE_COND, *(ROOT / p for p in wd.CODE), Path(__file__)])
    cm.check_generator(base["generator"])
    lex_path = ROOT / base["lexicon"]["file"]
    lex_raw = lex_path.read_bytes()
    if hashlib.sha256(lex_raw).hexdigest() != base["lexicon"]["sha256"]:
        raise SystemExit("the lexicon does not match the SHA-256 of the conditions")
    lex = load_lexicon(lex_path)
    # only the words are read from the record of version 3; its V and A are null in the archive, so its SHA-256 is not
    # the words_record_sha256 of the conditions, and the repeated run below checks the words through the figures
    rec3 = json.loads(wd.V3_RECORD.read_text())
    inputs = (cd, v3, base, rec3, lex, wd.read_vad(lex_raw))
    state = cm.git_state(["works", "src", "checks", "tools", "results/sketch_translations_words_d",
                          "results/sketch_translations_words", "pyproject.toml", "uv.lock", ".python-version"])
    if state["git_dirty"]:
        raise SystemExit("the tracked inputs differ from the last commit")
    recorded = json.loads(RECORD.read_text())["figures"]
    as_is = run_once(inputs, unlabel=False)
    for k, b in as_is.items():
        if hashlib.sha256(b).hexdigest() != recorded[k]:
            raise SystemExit(f"the repeated run does not give the recorded {k}")
    new = run_once(inputs, unlabel=True)
    if new["pairs.png"] != as_is["pairs.png"]:
        raise SystemExit("pairs.png changed although only the labels of words_vad.png were to change")
    changed = only_labels_removed(as_is["words_vad.png"], new["words_vad.png"], v3["tiles"]["rows"])
    rec = {"meta": {"conditions_sha256": hashlib.sha256(raw_c).hexdigest(), "lexicon_sha256": lex.sha256,
                    "atact_library_sha256": cm.library_sha256(), "uv_lock_sha256": cm.sha256(ROOT / "uv.lock"),
                    **cm.versions("numpy", "pillow"), **state,
                    "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
           "repeated_run_matches_record": {k: recorded[k] for k in as_is},
           "pixels_changed_all_white": changed,
           "figure": {"file": OUT_PNG.name, "sha256": hashlib.sha256(new["words_vad.png"]).hexdigest()}}
    text = json.dumps(rec, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
    with OUT_PNG.open("xb") as fh:
        fh.write(new["words_vad.png"])
    try:
        with OUT_JSON.open("x") as fh:
            fh.write(text)
    except BaseException:
        OUT_PNG.unlink()
        raise
    print(json.dumps(rec["figure"]))


if __name__ == "__main__":
    main()
