"""The list of the external inputs (external/, not in this archive): every path under external/ named by the scripts
and conditions (not the Aozora Bunko files that checks/reference_stretch.py fetches itself; their SHA-256 are in its
results), with its SHA-256, size and source. Run where external/ is present; writes external-manifest.tsv.

Usage:  uv run python tools/external_manifest.py <directory holding external/>
"""

from __future__ import annotations

import csv
import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "external-manifest.tsv"
AOZORA_LIST = "external/text-candidates-20261027/src/list_person_all_extended_utf8.csv"
SOURCES = {
    "external/nrc-vad-v2.1-20261026/NRC-VAD-Lexicon-v2.1/NRC-VAD-Lexicon-v2.1.txt":
        "NRC VAD Lexicon v2.1, https://saifmohammad.com/WebPages/nrc-vad.html (not redistributable)",
    "external/memolon-20261027/MTL_grouped-en.tsv":
        "MEmoLon, en.tsv of MTL_grouped.zip, https://doi.org/10.5281/zenodo.3756607 (CC BY 4.0)",
    "external/memolon-20261027/MTL_grouped-ja.tsv":
        "MEmoLon, ja.tsv of MTL_grouped.zip, https://doi.org/10.5281/zenodo.3756607 (CC BY 4.0)",
    AOZORA_LIST: "Aozora Bunko, list_person_all_extended_utf8.csv of "
                 "https://www.aozora.gr.jp/index_pages/list_person_all_extended_utf8.zip",
    "external/text-candidates-20261027/gutenberg/pg160.txt": "Project Gutenberg #160, "
                                                             "https://www.gutenberg.org/cache/epub/160/pg160.txt",
    "external/function-words-20261028/nltk-stopwords-english.txt":
        "stopwords/english of https://raw.githubusercontent.com/nltk/nltk_data/"
        "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a/packages/corpora/stopwords.zip",
    "external/translations-20261028/little-mermaid/paull-mermaid.txt": "Wikisource, revision 12188521, text of "
        "https://en.wikisource.org/wiki/Hans_Andersen%27s_Fairy_Tales/The_Little_Mermaid",
    "external/translations-20261028/little-mermaid/braekstad-mermaid.txt": "Wikisource, revision 10420654, text of "
        "https://en.wikisource.org/wiki/Fairy_tales_and_stories_(Andersen,_Tegner)/The_Little_Mermaid",
    "external/translations-20261028/little-mermaid/craigie-mermaid.txt": "Wikisource, revision 11193950, text of "
        "https://en.wikisource.org/wiki/Fairy_Tales_and_Other_Stories_(Andersen,_Craigie)/The_Little_Sea_Maid",
}
PATH = re.compile(r"external/[A-Za-z0-9._/-]+\.(?:txt|tsv|csv|zip)")


def named_paths() -> list[str]:
    files = subprocess.run(["git", "-C", str(ROOT), "ls-files", "works", "checks", "experiments", "tools", "src", "tests"],
                           capture_output=True, text=True, check=True).stdout.split()
    found = set()
    for f in files:
        if f.endswith((".py", ".json")):
            found |= set(PATH.findall((ROOT / f).read_text(encoding="utf-8")))
    return sorted(found)


def main(base: Path) -> None:
    aozora = {}
    with (base / AOZORA_LIST).open(encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if row and row[45].startswith("http"):
                aozora[row[45].rsplit("/", 1)[1]] = row[45]
    rows = []
    for p in named_paths():
        f = base / p
        if not f.exists():
            raise SystemExit(f"{p} is named but not present under {base}")
        src = SOURCES.get(p) or (f"Aozora Bunko, {aozora[f.name]}" if f.name in aozora else None)
        if src is None:
            raise SystemExit(f"no source written for {p}")
        rows.append((p, hashlib.sha256(f.read_bytes()).hexdigest(), str(f.stat().st_size), src))
    with OUT.open("w", encoding="utf-8") as fh:
        fh.write("path\tsha256\tbytes\tsource\n")
        fh.writelines("\t".join(r) + "\n" for r in rows)
    print(f"{len(rows)} files written to {OUT.name}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]).expanduser().resolve())
