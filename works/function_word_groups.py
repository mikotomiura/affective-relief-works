"""The groups (parts of speech) of the list of function words, from an external source only: for each word of the
list, the universal tag (Petrov, Das and McDonald 2012, the map en-brown.map of NLTK) that the word most often has in
the tagged Brown corpus of NLTK, matched in lower case. Words tagged ADP, CONJ, DET or PRON form those four groups;
every other word, and a word absent from Brown, forms the group "other". Ties go to the tag first in Unicode order.

Usage:  uv run python works/function_word_groups.py
Input:  external/function-words-20261028/{nltk-stopwords-english.txt,brown.zip,universal_tagset.zip} (not in the
        repository), checked against the SHA-256 written below
Output: the groups as JSON on standard output (copied into works/conditions/sketch_translations_style.json, whose
        script checks that they cover the list). No text of a translation is read.
"""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "external" / "function-words-20261028"
SHA = {"nltk-stopwords-english.txt": "f6d005956f407dbc6ea32e5ff0c7e8e6f71488d3239b9023efdc7fc139d6375b",
       "brown.zip": "9b275f9b3b95d7bd66ccfb7cd259f445a13bbe5d1f4107aba09fd3e8364bafa6",
       "universal_tagset.zip": "d490e1ae8f5625dcdfdda04be15c22a2aade8c2561a36a61edcdf0c7d6aa8352"}
KEPT = ("ADP", "CONJ", "DET", "PRON")


def check(name: str) -> bytes:
    raw = (DIR / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA[name]:
        sys.exit(f"{name} does not match its SHA-256")
    return raw


def universal(tag: str, table: dict[str, str]) -> str | None:
    """The universal tag of a Brown tag: as written, else without the suffixes -TL, -HL, -NC (title, headline,
    cited); None if still unknown."""
    t = tag.upper()
    while t not in table:
        for suf in ("-TL", "-HL", "-NC"):
            if t.endswith(suf):
                t = t[: -len(suf)]
                break
        else:
            return None
    return table[t]


def main() -> None:
    words = [w.strip() for w in check("nltk-stopwords-english.txt").decode("utf-8").splitlines() if w.strip()]
    check("universal_tagset.zip")
    with zipfile.ZipFile(DIR / "universal_tagset.zip") as z:
        table = dict(line.split("\t") for line in z.read("universal_tagset/en-brown.map").decode("utf-8").splitlines()
                     if line.strip())
    check("brown.zip")
    want = set(words)
    tags: dict[str, Counter] = defaultdict(Counter)
    unknown = Counter()
    with zipfile.ZipFile(DIR / "brown.zip") as z:
        for name in z.namelist():
            base = name.rsplit("/", 1)[-1]
            if len(base) != 4 or not base[:2].isalpha():  # the 500 text files ca01 .. cr09
                continue
            for tok in z.read(name).decode("latin-1").split():
                w, _, tag = tok.rpartition("/")
                if w.lower() in want:
                    u = universal(tag, table)
                    if u is None:
                        unknown[tag] += 1
                    else:
                        tags[w.lower()][u] += 1
    groups = {g: [] for g in (*KEPT, "other")}
    detail = {}
    for w in words:
        if tags[w]:
            top = sorted(tags[w].items(), key=lambda x: (-x[1], x[0]))[0][0]
            share = tags[w][top] / sum(tags[w].values())
        else:
            top, share = None, None
        groups[top if top in KEPT else "other"].append(w)
        detail[w] = {"tag": top, "share_of_top": share, "count": sum(tags[w].values())}
    print(json.dumps({"groups": groups, "detail": detail, "unknown_brown_tags": dict(unknown)}, indent=1))


if __name__ == "__main__":
    main()
