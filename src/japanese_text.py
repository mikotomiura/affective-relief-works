"""Japanese text step from a sentence to a point (V, A) in [-1, 1]^2, written for the works made with the rule.

It has the form of the English step of the generator (atact.lexicon): for each sentence, the words found in a lexicon
are averaged component by component. What differs, and why:

- Lexicon: the Japanese lexicon of MEmoLon (Buechel, Ruecker & Hahn 2020; predicted values on the nominal 1-to-9
  scale of Warriner et al., 5 neutral, values may lie outside). Each word value is mapped to [-1, 1] by a straight line
  per dimension and then clipped (the map is fixed elsewhere and passed in); the clipped values are averaged, so the
  mean lies in [-1, 1] and the sentence is not clipped again.
- Words: the sentence is cut by MeCab with IPAdic (fugashi), and each token is looked up by its surface form as
  written (no lemmatisation, no Unicode normalisation; the lexicon has entries for both full-width and half-width
  forms). Only tokens whose IPAdic part of speech is in POS_ALLOWED are averaged in the main form; the form "all"
  averages every token found in the lexicon except symbols (the English tokenizer never makes punctuation a word).
  Each occurrence counts, as in the English step.
- If no token is averaged (Z = 0) the sentence is given (0, 0) and flagged as unmatched; the number of candidate
  tokens (those that passed the part of speech) is kept with the number matched, so that coverage can be reported.
- Text from Aozora Bunko is cleaned first (strip_aozora) and cut into sentences (split_sentences), following the
  notation of the Aozora Bunko annotation list (https://www.aozora.gr.jp/annotation/).

The result is the mean of dictionary values of the matched words, not a sentence-level emotion detector: negation,
inflection and context are not handled. An inflected word is matched only if its surface form as cut by MeCab is itself
an entry of the lexicon (for example 嬉しく in 嬉しくない); it is counted as a candidate either way. IPAdic may tag a
common word as a proper noun in some contexts (seen for 毎日 and for the Latin word cafe), and the part of speech then
leaves it out.

The lexicon file is not part of this repository; its path and SHA-256 are recorded by each computation that uses it.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import fugashi
import ipadic

# IPAdic parts of speech that are averaged in the main form: (first level, second level). Everything else is left out,
# including proper nouns, pronouns, numbers, non-independent and suffix nouns and verbs, and all function words.
POS_ALLOWED = {
    "名詞": frozenset({"一般", "サ変接続", "形容動詞語幹", "副詞可能", "ナイ形容詞語幹"}),
    "動詞": frozenset({"自立"}),
    "形容詞": frozenset({"自立"}),
    "副詞": frozenset({"一般", "助詞類接続"}),
}
SYMBOL = "記号"
GAIJI = "〓"  # stands in for a character given by an external character note, so that its neighbours stay apart

SENTENCE_END = frozenset("。！？")
CLOSERS = frozenset("」』）〕】〉”’")
OPENERS = frozenset("「『（〔【〈“‘")
QUOTE_CLOSERS = frozenset("」』")


# ---------------------------------------------------------------- lexicon and map


@dataclass(frozen=True)
class Lexicon:
    entries: dict[str, tuple[float, float]]  # word -> (valence, arousal) on the scale of the file
    sha256: str = ""


def load_memolon(path: Path) -> Lexicon:
    """Read a MEmoLon tsv (word, valence, arousal, ...) with a header line; the hash is taken from the parsed bytes."""
    raw = Path(path).read_bytes()
    rows = csv.reader(io.StringIO(raw.decode("utf-8"), newline=""), delimiter="\t", quoting=csv.QUOTE_NONE)
    head = next(rows)
    if head[:3] != ["word", "valence", "arousal"]:
        raise ValueError(f"unexpected header in {path}: {head}")
    entries: dict[str, tuple[float, float]] = {}
    for i, r in enumerate(rows, 2):
        if len(r) < 3 or not r[0]:
            raise ValueError(f"line {i} of {path} has too few fields or an empty word")
        if r[0] != r[0].strip():
            raise ValueError(f"word with surrounding white space on line {i} of {path}: {r[0]!r}")
        if r[0] in entries:
            raise ValueError(f"duplicate entry {r[0]!r} in {path}")
        v, a = float(r[1]), float(r[2])
        if not (math.isfinite(v) and math.isfinite(a)):
            raise ValueError(f"non-finite value on line {i} of {path}")
        entries[r[0]] = (v, a)
    return Lexicon(entries, hashlib.sha256(raw).hexdigest())


class ValueMap(Protocol):
    """Maps a word value (valence, arousal) on the scale of the lexicon to [-1, 1]^2."""

    def __call__(self, v: float, a: float) -> tuple[float, float]: ...


@dataclass(frozen=True)
class LinearMap:
    """x' = b x + a per dimension, then clipped to [-1, 1]."""

    b_v: float
    a_v: float
    b_a: float
    a_a: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(x) for x in (self.b_v, self.a_v, self.b_a, self.a_a)):
            raise ValueError(f"non-finite coefficient in {self}")

    def raw(self, v: float, a: float) -> tuple[float, float]:
        return self.b_v * v + self.a_v, self.b_a * a + self.a_a

    def __call__(self, v: float, a: float) -> tuple[float, float]:
        x, y = self.raw(v, a)
        return clip(x), clip(y)


def clip(x: float) -> float:
    if not math.isfinite(x):  # min/max would turn NaN into a number, and an overflow into a bound
        raise ValueError(f"non-finite value {x!r}")
    return min(1.0, max(-1.0, x))


# ---------------------------------------------------------------- tokens


@dataclass(frozen=True)
class Token:
    surface: str
    pos1: str
    pos2: str


class Tagger:
    """MeCab with the IPAdic of the ipadic package (the SHA-256 of every file of its dictionary directory is exposed
    for the record: sys.dic, but also char.bin, unk.dic, matrix.bin and dicrc change how text is cut)."""

    def __init__(self) -> None:
        self._t = fugashi.GenericTagger(ipadic.MECAB_ARGS)
        self.dicdir = Path(ipadic.DICDIR)
        self.dicdir_sha256 = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(self.dicdir.iterdir()) if p.is_file()}

    def tokens(self, sentence: str) -> list[Token]:
        return [Token(w.surface, w.feature[0], w.feature[1]) for w in self._t(sentence)]


SELECTIONS = ("pos", "all")
ORDERS = ("word", "sentence")


def is_candidate(t: Token, select: str) -> bool:
    if select == "pos":
        return t.pos2 in POS_ALLOWED.get(t.pos1, ())
    if select == "all":
        return t.pos1 != SYMBOL
    raise ValueError(f"unknown selection {select!r}")


# ---------------------------------------------------------------- sentence point


@dataclass
class SentencePoint:
    text: str
    V: float
    A: float
    Z: int  # tokens matched in the lexicon and averaged
    n_candidates: int  # tokens that passed the selection, matched or not
    matched: list[str] = field(default_factory=list)

    @property
    def unmatched(self) -> bool:
        return self.Z == 0


def check_options(select: str, order: str) -> None:
    if select not in SELECTIONS or order not in ORDERS:
        raise ValueError(f"unknown selection {select!r} or order {order!r}")


def sentence_point(text: str, lex: Lexicon, tagger: Tagger, vmap: ValueMap, select: str = "pos",
                   order: str = "word") -> SentencePoint:
    """order="word": map and clip each word value, then average (the main form). order="sentence": average the values
    on the scale of the lexicon, then map and clip the mean (kept only for the sensitivity check)."""
    check_options(select, order)  # checked first, so that unmatched sentences do not hide a wrong option
    cand = [t.surface for t in tagger.tokens(text) if is_candidate(t, select)]
    matched = [w for w in cand if w in lex.entries]
    z = len(matched)
    if z == 0:
        return SentencePoint(text, 0.0, 0.0, 0, len(cand), [])
    if order == "word":
        pts = [vmap(*lex.entries[w]) for w in matched]
        V = sum(p[0] for p in pts) / z
        A = sum(p[1] for p in pts) / z
    else:
        V, A = vmap(sum(lex.entries[w][0] for w in matched) / z, sum(lex.entries[w][1] for w in matched) / z)
    return SentencePoint(text, V, A, z, len(cand), matched)


def text_path(sentences: list[str], lex: Lexicon, tagger: Tagger, vmap: ValueMap, select: str = "pos",
              order: str = "word") -> list[SentencePoint]:
    """The path of a text: one point per sentence, in order (unmatched sentences stay in place, flagged)."""
    check_options(select, order)
    return [sentence_point(s, lex, tagger, vmap, select, order) for s in sentences]


# ---------------------------------------------------------------- Aozora Bunko text

HYPHEN_LINE = re.compile(r"^-{10,}$")
RUBY = re.compile(r"《[^《》]*》")
FORWARD_HEADING = re.compile(r"［＃「([^「」［］]+)」は(?:同行|窓)?[大中小]見出し］")
ACCENT = re.compile(r"〔([\x20-\x7e]+)〕")  # accent decomposition: Latin letters and ASCII marks, closed on the line
INLINE_HEADING = re.compile(r"［＃((?:同行|窓)?[大中小]見出し)］.*?［＃\1終わり］")
BLOCK_HEADING_START = re.compile(r"［＃ここから((?:同行|窓)?[大中小]見出し)］")
HEADING_NOTE_END = re.compile(r"見出し(?:終わり)?］$")
HEADER_SEARCH_LINES = 60


@dataclass
class AozoraReport:
    lines_in: int = 0
    header_lines: int = 0
    footer_lines: int = 0
    footer_mark: str = ""
    ruby: int = 0
    headings: int = 0
    heading_block_lines: int = 0
    gaiji: int = 0
    notes: int = 0


def _note_end(s: str, i: int) -> int:
    """Index just after the note that opens at s[i] ("［＃"), counting nested brackets."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "［":
            depth += 1
        elif s[j] == "］":
            depth -= 1
            if depth == 0:
                return j + 1
    raise ValueError(f"unclosed note in line: {s!r}")


def _remove_notes(s: str, rep: AozoraReport, heading_guard: bool = True) -> str:
    """Replace each external character note ※［＃...］ by GAIJI and remove every other note ［＃...］."""
    out, i = [], 0
    while i < len(s):
        k = s.find("［＃", i)
        if k < 0:
            out.append(s[i:])
            break
        end = _note_end(s, k)
        if heading_guard and HEADING_NOTE_END.search(s[k:end]):
            # a heading note that the heading rules did not take (its text would stay in the body)
            raise ValueError(f"unhandled heading note: {s[k:end]!r}")
        if k > i and s[k - 1] == "※":
            out.append(s[i : k - 1] + GAIJI)
            rep.gaiji += 1
        else:
            out.append(s[i:k])
            rep.notes += 1
        i = end
    return "".join(out)


def _remove_forward_headings(s: str, rep: AozoraReport) -> str:
    """X［＃「X」は中見出し］ -> "" (the text before the note must end with X, which carries no ruby)."""
    while m := FORWARD_HEADING.search(s):
        x = m.group(1)
        if not s[: m.start()].endswith(x):
            raise ValueError(f"heading note does not follow its text: {s!r}")
        s = s[: m.start() - len(x)] + s[m.end() :]
        rep.headings += 1
    return s


def strip_aozora(text: str) -> tuple[str, AozoraReport]:
    """Body text of an Aozora Bunko file (already decoded), one paragraph per line.

    Removes, in this order: the header up to the second line of hyphens (title, author and the notes on symbols;
    an error if it is not found), the footer from the line that starts with 底本： or from the note ［＃本文終わり］,
    whichever comes first (an error if neither is found), the ruby start mark ｜ and ruby 《...》, headings (forward notes X［＃「X」は中見出し］, inline
    start/end notes and multi-line ［＃ここから中見出し］ blocks, with their text), external character notes
    ※［＃...］ (replaced by GAIJI), every other note ［＃...］ (nested brackets are matched; the text of a warichu note
    stays in the body), the brackets 〔〕 of accent decomposition (only around ASCII text, so an ordinary 〔1〕 loses
    its brackets too; other 〔〕 are kept as ordinary brackets), and leading and trailing white space of each line. Empty lines are kept as empty strings."""
    rep = AozoraReport()
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    rep.lines_in = len(lines)
    hy = [i for i, ln in enumerate(lines[:HEADER_SEARCH_LINES]) if HYPHEN_LINE.match(ln.strip())]
    if len(hy) < 2:
        raise ValueError("no header block between two lines of hyphens near the start")
    start = hy[1] + 1
    teihon = next((i for i in range(start, len(lines)) if lines[i].startswith("底本：")), None)
    owari = next((i for i in range(start, len(lines)) if "［＃本文終わり］" in lines[i]), None)
    if teihon is None and owari is None:
        raise ValueError("no footer (底本： or ［＃本文終わり］) after the header")
    if owari is not None and (teihon is None or owari < teihon):
        # the body ends at the note; text before the note on its line is kept
        end, rep.footer_mark = owari + 1, "本文終わり"
        lines = lines[:owari] + [lines[owari].split("［＃本文終わり］", 1)[0]] + lines[owari + 1 :]
    else:
        end, rep.footer_mark = teihon, "底本"
    rep.header_lines, rep.footer_lines = start, len(lines) - end
    out: list[str] = []
    block: str | None = None
    for ln in lines[start:end]:
        if block is not None:
            rep.heading_block_lines += 1
            if f"［＃ここで{block}終わり］" in ln:
                block = None
                _no_text_outside_notes(ln)
            continue
        if m := BLOCK_HEADING_START.search(ln):
            _no_text_outside_notes(ln)
            if f"［＃ここで{m.group(1)}終わり］" in ln:
                raise ValueError(f"heading block opens and closes on one line: {ln!r}")
            block = m.group(1)
            rep.headings += 1
            rep.heading_block_lines += 1
            continue
        n = len(RUBY.findall(ln))
        rep.ruby += n
        ln = RUBY.sub("", ln).replace("｜", "")
        ln = _remove_forward_headings(ln, rep)
        ln, k = INLINE_HEADING.subn("", ln)
        rep.headings += k
        ln = _remove_notes(ln, rep)
        ln = ACCENT.sub(r"\1", ln)
        out.append(ln.strip())
    if block is not None:
        raise ValueError(f"heading block {block} is not closed")
    return "\n".join(out), rep


def _no_text_outside_notes(ln: str) -> None:
    """A line that opens or closes a multi-line heading block must hold nothing but notes (it is dropped whole)."""
    if _has_word_char(_remove_notes(ln, AozoraReport(), heading_guard=False)):
        raise ValueError(f"text beside a heading block note: {ln!r}")


def _has_word_char(s: str) -> bool:
    return any(unicodedata.category(ch)[0] in "LN" for ch in s)  # GAIJI (〓, So) is not a letter


def _next_visible(s: str, i: int) -> str:
    while i < len(s) and s[i].isspace():
        i += 1
    return s[i] if i < len(s) else ""


def split_line(line: str) -> list[str]:
    """Sentences of one line (see _pieces); pieces without a letter or a digit (Unicode L or N) are not sentences."""
    return [p for p in _pieces(line) if _has_word_char(p)]


def _pieces(line: str) -> list[str]:
    """Sentences of one line. A sentence ends after 。！？ (with any following run of these marks and closing
    brackets), or after a run of closing brackets that contains 」 or 』 when the run ends the line or the next visible
    character opens a bracket. Empty pieces are left out."""
    out, cur, i, n = [], [], 0, len(line)
    while i < n:
        ch = line[i]
        cur.append(ch)
        i += 1
        if ch in SENTENCE_END:
            while i < n and (line[i] in SENTENCE_END or line[i] in CLOSERS):
                cur.append(line[i])
                i += 1
            out.append("".join(cur))
            cur = []
        elif ch in CLOSERS:
            run = [ch]
            while i < n and line[i] in CLOSERS:
                run.append(line[i])
                cur.append(line[i])
                i += 1
            if QUOTE_CLOSERS & set(run) and (i == n or _next_visible(line, i) in OPENERS):
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return [s.strip() for s in out if s.strip()]


def split_sentences(body: str) -> tuple[list[str], list[str]]:
    """Sentences of a body text: every line break ends a sentence, then split_line within each line. Also returns the
    pieces that were not sentences (no letter or digit, e.g. ＊　＊　＊), so that the record can count them."""
    kept: list[str] = []
    dropped: list[str] = []
    for ln in body.split("\n"):
        for piece in _pieces(ln):
            (kept if _has_word_char(piece) else dropped).append(piece)
    return kept, dropped
