"""The text steps of the two works, from a text file to sentence points (work 1) or to window trajectories (work 2).

Work 1 (English): extract_story -> split_sentences (src/english_text.py) -> the generator's sentence_point
(atact.lexicon: every matched term averaged, an unmatched sentence is (0, 0) and flagged) -> (f, s, gamma).
Work 2 (Japanese): strip_aozora -> strip_date_note -> split_sentences -> sentence_point (src/japanese_text.py) ->
trajectory (src/window.py, W = 40) -> stretch (src/stretch.py, the factor fixed beforehand). Nothing is drawn here:
the portrait takes the trajectories (draw is a slot for the form that is still to be decided).
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from atact.lexicon import Lexicon as EnglishLexicon
from atact.lexicon import sentence_point as english_point
from atact.mapping import params

import english_text
import japanese_text
import stretch as stretch_mod
import window


@dataclass
class Work1Text:
    report: english_text.StoryReport
    sentences: list[str]
    dropped: list[str]
    points: list[dict]


def work1_text(raw: str, heading: str, next_heading: str | None, occurrence: int, lex: EnglishLexicon) -> Work1Text:
    paras, rep = english_text.extract_story(raw, heading, next_heading, occurrence)
    kept, dropped = english_text.split_sentences(paras)
    pts = []
    for s in kept:
        p = english_point(s, lex)
        f, sv, g = (float(x) for x in params(p.V, p.A))
        pts.append({"text": s, "V": p.V, "A": p.A, "Z": p.Z, "unmatched": p.unmatched, "matched": p.matched,
                    "f": f, "s": sv, "gamma": g})
    return Work1Text(rep, kept, dropped, pts)


def read_aozora_zip(path: Path) -> tuple[str, bytes]:
    """The one .txt member of an Aozora Bunko zip (an error if there is not exactly one)."""
    with zipfile.ZipFile(io.BytesIO(Path(path).read_bytes())) as z:
        txt = [n for n in z.namelist() if n.lower().endswith(".txt")]
        if len(txt) != 1:
            raise ValueError(f"{path} holds {len(txt)} .txt members")
        return txt[0], z.read(txt[0])


@dataclass
class Work2Text:
    report: japanese_text.AozoraReport
    date_note: str | None
    body_sha256: str
    sentences: list[str]
    dropped: list[str]
    points: list[japanese_text.SentencePoint]
    trajectory: list[window.WindowPoint]


def work2_body(text: str) -> tuple[str, japanese_text.AozoraReport, str | None, list[str], list[str]]:
    """Text step without the lexicon (the counts of the selection use this part only)."""
    body, rep = japanese_text.strip_aozora(text)
    body, note = japanese_text.strip_date_note(body)
    kept, dropped = japanese_text.split_sentences(body)
    return body, rep, note, kept, dropped


def work2_text(text: str, lex: japanese_text.Lexicon, tagger: japanese_text.Tagger, vmap: japanese_text.ValueMap,
               w: int = window.W_WORK2) -> Work2Text:
    body, rep, note, kept, dropped = work2_body(text)
    pts = japanese_text.text_path(kept, lex, tagger, vmap, select="pos", order="word")
    traj = window.trajectory([window.Sentence(p.V, p.A, not p.unmatched) for p in pts], w)
    return Work2Text(rep, note, hashlib.sha256(body.encode("utf-8")).hexdigest(), kept, dropped, pts, traj)


def window_record(p: window.WindowPoint) -> dict:
    rec = {"start": p.start, "position": p.position, "n_matched": p.n_matched, "gap": p.gap, "V": p.V, "A": p.A}
    if not p.gap:
        rec.update(zip(("f", "s", "gamma"), (float(x) for x in params(p.V, p.A))))
    return rec


def work2_trajectories(traj: list[window.WindowPoint], k: float) -> dict:
    """The unstretched and the stretched trajectory of one work, with the number of clipped windows."""
    st = stretch_mod.stretch(traj, k)
    return {"unstretched": [window_record(p) for p in traj], "stretched": [window_record(p) for p in st.points],
            "clipped": st.clipped, "with_point": st.with_point}


# The form of the portrait is not decided (.steering/20261028-while-waiting/design.md, P1-P3). A drawing function takes
# the trajectories of one author and an output directory, and returns the files it wrote.
Draw = Callable[[dict, Path], list[Path]]


def calibration_map(raw: bytes) -> japanese_text.LinearMap:
    """Map B from the bytes of results/calibrate_memolon.json (b and a of valence and arousal)."""
    d = json.loads(raw)["result"]["dims"]
    return japanese_text.LinearMap(d["valence"]["b"], d["valence"]["a"], d["arousal"]["b"], d["arousal"]["a"])
