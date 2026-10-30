"""Leave out, and put back, the values of the NRC VAD Lexicon v2.1 in the two records of the sketch of a series of
translators (conditions: tools/conditions/lexicon_values.json, committed before this tool).

Usage:
  uv run python tools/lexicon_values.py strip <original repository>   writes the two records with the values set to
                                                                        null (refuses to overwrite)
  uv run python tools/lexicon_values.py restore                         puts the values back from the lexicon in
                                                                        external/ and overwrites the two records only
                                                                        if the result has the original SHA-256
                                                                        (if it stops while writing, run it again: a
                                                                        record already restored is skipped)

The lexicon is not in this archive (its terms of use forbid redistribution); restore needs your own copy at the path
named in the conditions. Order of use when the archive was prepared: strip, commit the two records, then
tools/words_vad_unlabelled.py.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.pycache_prefix = tempfile.mkdtemp(prefix="pycache-")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "works"))
sys.path.insert(0, str(ROOT / "checks"))

COND = ROOT / "tools" / "conditions" / "lexicon_values.json"


def dump(d: dict) -> bytes:
    """The serialisation of the records (works/sketch_translations_words*.py)."""
    return (json.dumps(d, indent=1, ensure_ascii=False, allow_nan=False) + "\n").encode()


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def words_of(d: dict) -> list[dict]:
    """The entries of translations.<name>.words[] of a record, in order."""
    out = []
    for name, t in d["translations"].items():
        if not isinstance(t.get("words"), list) or not t["words"]:
            raise SystemExit(f"translations.{name}.words is missing or empty")
        out += t["words"]
    return out


def strip(d: dict, nulled: list[str]) -> dict:
    s = copy.deepcopy(d)
    for w in words_of(s):
        for k in nulled:
            if k not in w:
                raise SystemExit(f"{w.get('word')!r} has no key {k!r}")
            w[k] = None
    return s


def check_stripped(orig: dict, s: dict, nulled: list[str]) -> None:
    """Compared with the original independently of strip: the same keys in the same order everywhere, every listed key
    of every word null, and every other value equal."""

    def walk(a, b, path: str, in_word: bool) -> None:
        if isinstance(a, dict):
            if not isinstance(b, dict) or list(a) != list(b):
                raise SystemExit(f"{path}: the keys differ")
            for k in a:
                if in_word and k in nulled:
                    if b[k] is not None:
                        raise SystemExit(f"{path}.{k}: a value was left in")
                else:
                    walk(a[k], b[k], f"{path}.{k}" if path else k, False)
        elif isinstance(a, list):
            if not isinstance(b, list) or len(a) != len(b):
                raise SystemExit(f"{path}: the lists differ")
            words = path.startswith("translations.") and path.endswith(".words")
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{i}]", words)
        elif type(a) is not type(b) or a != b:
            raise SystemExit(f"{path}: a value other than those listed differs")

    walk(orig, s, "", False)
    if len(words_of(s)) != len(words_of(orig)):
        raise SystemExit("the number of words differs")


def main_strip(src_root: Path) -> None:
    """Every file is read and checked, and every output path found free, before anything is written."""
    c = json.loads(COND.read_text())
    out = {}
    for f in c["files"]:
        src, dst = src_root / f["path"], ROOT / f["path"]
        raw = src.read_bytes()
        if sha(raw) != f["sha256_original"]:
            raise SystemExit(f"{src} is not the original named in the conditions")
        if dump(json.loads(raw)) != raw:
            raise SystemExit(f"{src} does not round-trip through the serialisation")
        if dst.exists():
            raise SystemExit(f"{dst} exists; refusing to overwrite")
        orig = json.loads(raw)
        s = strip(orig, f["nulled"])
        check_stripped(orig, s, f["nulled"])
        out[dst] = (f, dump(s), len(words_of(s)))
    for dst, (f, b, n) in out.items():
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("xb") as fh:
            fh.write(b)
        print(f"{f['path']}: {n} words, {', '.join(f['nulled'])} set to null; sha256 {sha(b)}")


def restore_words(s: dict, lex) -> dict:
    d = copy.deepcopy(s)
    for w in words_of(d):
        if w["word"] not in lex.entries:
            raise SystemExit(f"{w['word']!r} is not in the lexicon")
        w["V"], w["A"] = lex.entries[w["word"]][0], lex.entries[w["word"]][1]
    return d


def restore_words_d(s: dict, words_record: dict, lex, lex_raw: bytes) -> dict:
    """The rows of the sketch of a third input computed again by its own run (works/sketch_translations_words_d.py)
    in a temporary directory, on the restored record of version 3; every value kept in s must equal the rerun."""
    import _common as cm
    import sketch_translations_words_d as wd

    _, cd = cm.load_conditions(wd.COND)
    _, v3 = cm.load_conditions(wd.V3_COND)
    _, base = cm.load_conditions(wd.BASE_COND)
    cm.check_generator(base["generator"])
    with tempfile.TemporaryDirectory() as tmp:
        res = wd.run(cd, v3, base, words_record, lex, wd.read_vad(lex_raw), Path(tmp) / "out")
    if res["figures"] != s["figures"]:
        raise SystemExit("the rerun of the sketch does not give the figures of its record")
    rows = res["translations"]
    d = copy.deepcopy(s)
    d["translations"] = rows
    check_stripped(d, s, json.loads(COND.read_text())["files"][1]["nulled"])
    return d


def write_whole(path: Path, b: bytes) -> None:
    """Replaces the file whole, keeping its permissions: a stop leaves it as it was or as written, never in part."""
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as fh:
        tmp = Path(fh.name)
    try:
        tmp.write_bytes(b)
        os.chmod(tmp, path.stat().st_mode)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def main_restore() -> None:
    from atact.lexicon import load_lexicon

    c = json.loads(COND.read_text())
    lex_path = ROOT / c["lexicon"]["file"]
    if not lex_path.exists():
        raise SystemExit(f"{lex_path} not found; download the lexicon from {c['lexicon']['source']}")
    lex_raw = lex_path.read_bytes()
    if sha(lex_raw) != c["lexicon"]["sha256"]:
        raise SystemExit("the lexicon differs from the one named in the conditions")
    lex = load_lexicon(lex_path)
    words_f, words_d_f = c["files"]  # version 3 first: the sketch of a third input is computed on its words
    restored = {}
    words = json.loads((ROOT / words_f["path"]).read_bytes())
    if sha(dump(words)) != words_f["sha256_original"]:
        words = restore_words(words, lex)
    for f, compute in ((words_f, lambda s: restore_words(s, lex)),
                       (words_d_f, lambda s: restore_words_d(s, words, lex, lex_raw))):
        path = ROOT / f["path"]
        raw = path.read_bytes()
        if sha(raw) == f["sha256_original"]:
            print(f"{f['path']}: already the original")
            continue
        b = dump(compute(json.loads(raw)))
        if sha(b) != f["sha256_original"]:
            raise SystemExit(f"{f['path']}: the restored record differs from the original; nothing was written")
        restored[path] = b
    for path, b in restored.items():
        write_whole(path, b)
        print(f"{path.relative_to(ROOT)}: restored, sha256 {sha(b)} (the original)")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "strip":
        main_strip(Path(sys.argv[2]).expanduser().resolve())
    elif len(sys.argv) == 2 and sys.argv[1] == "restore":
        main_restore()
    else:
        raise SystemExit(__doc__)
