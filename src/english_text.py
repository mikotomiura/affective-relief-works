"""English text step from a Project Gutenberg plain-text file to a list of sentences, written for the works made with
the rule.

The pinned generator is used unchanged: its sentence_point (atact.lexicon) takes one sentence at a time. What is
defined here is only what a sentence is. The generator's own split_sentences cuts after . ! ? followed by white space;
on prose with dialogue this cuts after title abbreviations (Mrs. Smith) and does not cut where a sentence ends inside
a closing quotation mark (." He). The rule here (split_paragraph) differs from it on purpose, and the works state so.

Extraction (extract_story): the body of one story is the run of lines strictly between a heading line and the next
heading line (both matched exactly after stripping, both given by the caller), inside the START and END markers of
the Gutenberg file. Lines of a paragraph are joined by one space; paragraphs that are wholly in square brackets
(illustration notes such as [Picture: ...], nested brackets allowed: the first [ closes only at the last ]) are
dropped and listed. Underscores that mark italics are left in place
(the generator's tokenizer does not read them).
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field

START_MARK = "*** START OF"
END_MARK = "*** END OF"

# A sentence ends after a run of terminal marks (. ! ? and the ellipsis character), followed by any run of closing
# quotation marks or brackets, then white space (any Unicode space, the no-break space included), when the next
# visible character opens a new sentence. The straight quotation marks " and ' close and open alike.
TERMINALS = frozenset(".!?…")
CLOSERS = frozenset("”’\"')]")
OPENERS = frozenset("“‘’\"'([")  # ’ also opens an elided word at the start of a sentence (’Twas)
# Title abbreviations after which a single full stop does not end a sentence (matched as whole words, case as written)
ABBREVIATIONS = frozenset({"Mr", "Mrs", "Dr", "St", "Messrs", "Mme", "Mlle"})
_WORD_BEFORE = re.compile(r"([A-Za-z]+)$")


@dataclass
class StoryReport:
    start_line: int  # 1-based number of the heading line
    end_line: int  # 1-based number of the line that ends the body (the next heading or the END marker)
    paragraphs: int = 0
    dropped_brackets: list[str] = field(default_factory=list)
    body_sha256: str = ""


def extract_story(raw: str, heading: str, next_heading: str | None, occurrence: int = 0) -> tuple[list[str], StoryReport]:
    """Paragraphs of one story. heading is matched exactly (after stripping) and occurrence picks which match, counted
    from 0 among the matches inside the markers (the table of contents may hold the same line). The body ends at the
    first line after the heading that equals next_heading, or at the END marker if next_heading is None."""
    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    start = [i for i, ln in enumerate(lines) if ln.startswith(START_MARK)]
    end = [i for i, ln in enumerate(lines) if ln.startswith(END_MARK)]
    if len(start) != 1 or len(end) != 1 or end[0] <= start[0]:
        raise ValueError("expected one START and one END marker, in this order")
    lo, hi = start[0], end[0]
    heads = [i for i in range(lo + 1, hi) if lines[i].strip() == heading]
    if not -len(heads) <= occurrence < len(heads):
        raise ValueError(f"heading {heading!r} occurs {len(heads)} times; occurrence {occurrence} does not exist")
    s = heads[occurrence]
    if next_heading is None:
        e = hi
    else:
        e = next((i for i in range(s + 1, hi) if lines[i].strip() == next_heading), None)
        if e is None:
            raise ValueError(f"next heading {next_heading!r} not found after line {s + 1}")
    rep = StoryReport(s + 1, e + 1)
    paras: list[str] = []
    cur: list[str] = []
    for ln in lines[s + 1 : e] + [""]:
        if ln.strip():
            cur.append(ln.strip())
        elif cur:
            p = " ".join(cur)
            cur = []
            if _wholly_bracketed(p):
                rep.dropped_brackets.append(p)
            else:
                paras.append(p)
    if not paras:
        raise ValueError("the story has no paragraph")
    rep.paragraphs = len(paras)
    rep.body_sha256 = hashlib.sha256("\n".join(paras).encode("utf-8")).hexdigest()
    return paras, rep


def _wholly_bracketed(p: str) -> bool:
    """The paragraph starts with [ and ends with ], and the bracket opened first closes only at the end."""
    if not (p.startswith("[") and p.endswith("]")):
        return False
    depth = 0
    for i, ch in enumerate(p):
        depth += (ch == "[") - (ch == "]")
        if depth == 0 and i < len(p) - 1:
            return False
    return depth == 0


def _opens_sentence(ch: str) -> bool:
    return ch in OPENERS or ch.isdigit() or (unicodedata.category(ch) == "Lu")


def _after_abbreviation(text: str, run_start: int, run: str) -> bool:
    """A single full stop directly after a listed abbreviation (Mrs.) does not end a sentence."""
    if run != ".":
        return False
    m = _WORD_BEFORE.search(text[:run_start])
    return bool(m) and m.group(1) in ABBREVIATIONS and (m.start() == 0 or not text[m.start() - 1].isalnum())


def split_paragraph(p: str) -> list[str]:
    """Pieces of one paragraph, cut as described at TERMINALS, CLOSERS, OPENERS and ABBREVIATIONS. The end of the
    paragraph always ends a piece. Pieces without a letter or a digit are returned too; split_sentences sets them
    apart."""
    out, i, last, n = [], 0, 0, len(p)
    while i < n:
        if p[i] not in TERMINALS:
            i += 1
            continue
        j = i
        while j < n and p[j] in TERMINALS:
            j += 1
        run = p[i:j]
        k = j
        while k < n and p[k] in CLOSERS:
            k += 1
        m = k
        while m < n and p[m].isspace():
            m += 1
        if m > k and m < n and _opens_sentence(p[m]) and not _after_abbreviation(p, i, run):
            out.append(p[last:k].strip())
            last = m
        i = k if k > i else i + 1
    out.append(p[last:].strip())
    return [s for s in out if s]


def _has_word_char(s: str) -> bool:
    return any(unicodedata.category(ch)[0] in "LN" for ch in s)


def split_sentences(paragraphs: list[str]) -> tuple[list[str], list[str]]:
    """Sentences of a story, in order, and the pieces that were not sentences (no letter or digit, e.g. * * *)."""
    kept: list[str] = []
    dropped: list[str] = []
    for p in paragraphs:
        for s in split_paragraph(p):
            (kept if _has_word_char(s) else dropped).append(s)
    return kept, dropped
