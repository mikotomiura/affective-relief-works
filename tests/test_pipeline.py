"""End to end on self-written texts in the form of a Project Gutenberg file and of Aozora Bunko zips, with fake
lexicons of a few words (no text of any work and no real lexicon)."""

import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "works"))

import _common as cm  # noqa: E402
import japanese_text  # noqa: E402
import relief  # noqa: E402
import work1  # noqa: E402
import work2  # noqa: E402
from atact.lexicon import load_lexicon  # noqa: E402
from pipeline import calibration_map, work2_body  # noqa: E402

WORDS = ["happy", "sad", "calm", "storm", "home"]
NRC = "term\tvalence\tarousal\tdominance\nhappy\t0.9\t0.5\t0.1\nsad\t-0.8\t-0.3\t-0.2\ncalm\t0.4\t-0.9\t0.0\n" \
      "storm\t-0.3\t0.95\t0.2\nhome\t0.5\t-0.2\t0.1\n"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def gutenberg(n: int) -> str:
    sent = [f"The {WORDS[i % 5]} {'dog' if i % 3 else 'cat'} went to the {WORDS[(i * 2) % 5]} place {i}." for i in range(n)]
    sent[7] = "Mrs. Smith said nothing at all."  # no term, and no cut after the abbreviation
    paras = [" ".join(sent[i : i + 5]) for i in range(0, n, 5)]
    body = "\n\n".join(p[:40] + "\n" + p[40:] for p in paras)  # paragraphs over two lines
    return ("Header\n*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\nCONTENTS\n\nTHE TEST STORY\nTHE NEXT STORY\n\n"
            f"THE TEST STORY\n\n{body}\n\n[Illustration: a dog]\n\nTHE NEXT STORY\n\nOther text.\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\nFooter\n")


def work1_conditions(tmp: Path, n: int) -> dict:
    raw = gutenberg(n).encode("utf-8")
    (tmp / "pg.txt").write_bytes(raw)
    c = json.loads((ROOT / "works/conditions/work1.template.json").read_text())
    from english_text import extract_story, split_sentences

    paras, rep = extract_story(raw.decode(), "THE TEST STORY", "THE NEXT STORY", -1)
    c["text"].update(source="test", file="pg.txt", sha256=sha(raw), heading="THE TEST STORY",
                     next_heading="THE NEXT STORY", heading_line=rep.start_line, body_sha256=rep.body_sha256,
                     sentences=len(split_sentences(paras)[0]))
    c["fixed_before_running"] = "test"
    c["relief"].update(L_mm=5.0, grid_intervals=25)  # a small tile, so that the test is quick
    c["relief"]["eps"] = relief.eps_for(work1.dimensions(c))
    return c


def nrc(tmp: Path):
    (tmp / "nrc.txt").write_text(NRC)
    return load_lexicon(tmp / "nrc.txt")


def test_the_template_does_not_run():
    for t in ("work1", "work2"):
        p = ROOT / f"works/conditions/{t}.template.json"
        assert cm.placeholders(json.loads(p.read_text()))
        with pytest.raises(SystemExit, match="still to be filled"):
            cm.load_conditions(p)


def test_work1_end_to_end(tmp_path):
    c = work1_conditions(tmp_path, 45)
    assert c["text"]["sentences"] == 45
    lex = nrc(tmp_path)
    res = work1.run(c, tmp_path, lex, tmp_path / "out", tmp_path / "stl")
    pts = res["sentences"]
    assert len(pts) == 45 and res["unmatched"] == 1 and pts[7]["unmatched"] and pts[7]["V"] == 0.0
    assert pts[0]["matched"] == ["happy", "happy"] and pts[0]["V"] == pytest.approx(0.9)
    d = work1.dimensions(c)
    for i, p in enumerate(pts):
        for scale in relief.SCALES:
            s = p["stl"][scale]
            nm, tris = relief.read_stl(tmp_path / "stl" / s["file"])
            assert len(tris) == s["facets"]
            ec = relief.edge_check(tris)
            assert ec["not_two"] == 0 and ec["same_direction"] == 0
            v = tris.reshape(-1, 3)
            assert v[:, 0].min() == 0 and v[:, 0].max() == pytest.approx(d.L_mm)
            assert v[:, 2].min() == 0 and d.base_mm - 1e-6 <= v[v[:, 2] > 0][:, 2].min()
            assert v[:, 2].max() <= d.base_mm + d.D_mm + 1e-6
            assert relief.signed_volume(tris) > d.L_mm ** 2 * d.base_mm
    # the STL is the tile of the conditions: recompute one top from the point and compare
    p = pts[3]
    t = relief.tile(p["V"], p["A"], lambda f: relief.closed_field(f, 25, 42), c["relief"]["eps"])
    _, tris = relief.read_stl(tmp_path / "stl" / p["stl"]["common"]["file"])
    top = relief.top_surface(t.H_eps, "common", d)
    assert np.allclose(np.sort(tris[: 2 * 25 * 25, :, 2].ravel()),
                       np.sort(np.concatenate([top[:-1, :-1].ravel(), top[:-1, 1:].ravel(), top[1:, 1:].ravel(),
                                               top[:-1, :-1].ravel(), top[1:, 1:].ravel(), top[1:, :-1].ravel()])),
                       atol=1e-5)
    for name in ("common", "minmax", "common_eps"):
        im = Image.open(tmp_path / "out" / res["figures"][name]["file"])
        assert im.size[0] == round(177.8 / 25.4 * 300) and im.mode == "L"
    with pytest.raises(FileExistsError):
        work1.run(c, tmp_path, lex, tmp_path / "out", tmp_path / "stl")


@pytest.mark.parametrize("change,msg", [
    (lambda c: c["text"].update(sentences=44), "sentences"),
    (lambda c: c["text"].update(body_sha256="0" * 64), "extracted story"),
    (lambda c: c["text"].update(sha256="0" * 64), "SHA-256"),
    (lambda c: c["relief"].update(eps=0.3), "eps"),
    (lambda c: c["relief"].update(grid_intervals=24), "grid"),
    (lambda c: c["relief"].update(seed=True), "seed"),
    (lambda c: c["range"].update(max_sentences=44), "range"),
])
def test_work1_stops_on_wrong_conditions(tmp_path, change, msg):
    c = work1_conditions(tmp_path, 45)
    change(c)
    with pytest.raises(SystemExit, match=msg):
        work1.run(c, tmp_path, nrc(tmp_path), tmp_path / "out", tmp_path / "stl")
    assert not (tmp_path / "stl").exists()


# ---------------------------------------------------------------- work 2

JA_WORDS = {"嬉しい": (7.0, 5.0), "悲しい": (3.0, 4.0), "静か": (5.5, 3.0), "嵐": (4.0, 6.5), "家": (5.5, 4.0)}
ADJ = ["嬉しい", "悲しい", "静かな", "嵐の", "家の"]


def aozora_zip(path: Path, n: int, seed: int, unmatched_run: tuple[int, int] = (0, 0)) -> str:
    rng = np.random.default_rng(seed)
    lines = []
    for i in range(n):
        if unmatched_run[0] <= i < unmatched_run[1] or i % 7 == 6:
            lines.append("ああ。")  # no matched word
        else:
            lines.append(f"{ADJ[rng.integers(5)]}朝に私は{ADJ[rng.integers(5)]}道を歩いた。")
    text = ("試しの作品\n試しの作者\n\n" + "-" * 55 + "\n【テキスト中に現れる記号について】\n《》：ルビ\n" + "-" * 55
            + "\n\n" + "\n".join(lines[:3]) + "嵐《あらし》。\n" + "\n".join(lines[3:]) + "\n（大正六年十一月）\n\n\n\n"
            + "底本：試しの本\n")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("test_ruby.txt", text.encode("cp932"))
    return text


def work2_conditions(tmp: Path) -> dict:
    c = json.loads((ROOT / "works/conditions/work2.template.json").read_text())
    works = []
    for a in range(3):
        for k in range(3):
            n = 200 + 10 * k
            zp = tmp / f"a{a}w{k}.zip"
            text = aozora_zip(zp, n, 10 * a + k, (100, 130) if (a, k) == (0, 0) else (0, 0))
            body, _, note, kept, _ = work2_body(text)
            works.append({"author": f"author{a}", "work_id": f"{a}{k}", "title": "t", "zip": zp.name,
                          "zip_sha256": sha(zp.read_bytes()), "txt_name": "test_ruby.txt", "encoding": "cp932",
                          "body_sha256": sha(body.encode()), "sentences": len(kept), "date_note": note})
    c.update(works=works, fixed_before_running="test")
    c["stretch"].update(k=1.5, sha256="0" * 64)
    return c


@pytest.fixture(scope="module")
def ja():
    lex = japanese_text.Lexicon(JA_WORDS)
    vmap = calibration_map((ROOT / "results/calibrate_memolon.json").read_bytes())
    return lex, japanese_text.Tagger(), vmap


def test_work2_end_to_end(tmp_path, ja):
    c = work2_conditions(tmp_path)
    drawn = []

    def draw(author, out):
        drawn.append(author["author"])
        return [out / f"{author['author']}.txt"]

    res = work2.run(c, tmp_path, *ja, draw=draw, out_dir=tmp_path)
    assert sorted(res["authors"]) == ["author0", "author1", "author2"] and drawn == sorted(drawn)
    for a, ws in res["authors"].items():
        assert len(ws) == 3
        for w in ws:
            assert w["date_note"] == "（大正六年十一月）"
            assert w["sentences"] == len(w["unstretched"]) + 39
            assert 0 < w["matched_share"] < 1
            for u, s in zip(w["unstretched"], w["stretched"]):
                assert u["gap"] == s["gap"] and u["position"] == s["position"]
                if not u["gap"]:
                    assert s["V"] == pytest.approx(max(-1, min(1, 1.5 * u["V"])))
                    assert s["A"] == pytest.approx(max(-1, min(1, 1.5 * u["A"])))
                    assert {"f", "s", "gamma"} <= set(s)
    first = res["authors"]["author0"][0]
    assert any(p["gap"] for p in first["unstretched"])  # the unmatched run cuts the line
    assert not any(p["gap"] for p in res["authors"]["author1"][0]["unstretched"])


@pytest.mark.parametrize("change,msg", [
    (lambda c: c["works"][0].update(sentences=1), "range"),
    (lambda c: c["works"][0].update(body_sha256="0" * 64), "differs"),
    (lambda c: c["works"][0].update(zip_sha256="0" * 64), "SHA-256"),
    (lambda c: c["works"][0].update(txt_name="x.txt"), "holds"),
    (lambda c: c["works"][0].update(encoding="utf-8"), "cp932"),
    (lambda c: c["works"][1].update(work_id=c["works"][0]["work_id"]), "twice"),
    (lambda c: c.update(works=c["works"][:6]), "authors"),
    (lambda c: c.update(works=c["works"][:8]), "fewer works"),
    (lambda c: c["window"].update(W=30), "window"),
    (lambda c: c["stretch"].update(k=-1.0), "stretch"),
    (lambda c: c["step"].update(select="all"), "step"),
])
def test_work2_stops_on_wrong_conditions(tmp_path, ja, change, msg):
    c = work2_conditions(tmp_path)
    change(c)
    with pytest.raises(SystemExit, match=msg):
        work2.run(c, tmp_path, *ja)


def test_work2_window_value_is_the_mean_of_the_sentence_points(tmp_path, ja):
    lex, tagger, vmap = ja
    c = work2_conditions(tmp_path)
    res = work2.run(c, tmp_path, *ja)
    w = c["works"][3]  # author1, first work
    from pipeline import read_aozora_zip

    _, raw = read_aozora_zip(tmp_path / w["zip"])
    kept = work2_body(raw.decode("cp932"))[3]
    pts = [japanese_text.sentence_point(s, lex, tagger, vmap, "pos", "word") for s in kept[:40]]
    m = [p for p in pts if not p.unmatched]
    got = res["authors"]["author1"][0]["unstretched"][0]
    assert got["n_matched"] == len(m)
    assert got["V"] == pytest.approx(sum(p.V for p in m) / len(m))
    assert got["A"] == pytest.approx(sum(p.A for p in m) / len(m))
    # word order: each word value mapped and clipped before the mean (the sentence order would differ here)
    # (嵐 has A 6.5, which map B takes past 1; a sentence with it and another word separates the two orders)
    s = next(x for x in kept if "嵐" in x and "家" in x)
    word, sent = (japanese_text.sentence_point(s, lex, tagger, vmap, "pos", o) for o in ("word", "sentence"))
    assert word.A != pytest.approx(sent.A)


def test_placeholders_with_text_after_the_mark():
    assert cm.placeholders({"a": "FILL: the date", "b": ["x", {"c": "FILL"}], "d": "filled"}) == [".a", ".b[1].c"]


def test_matches_template():
    t = {"a": "FILL: x", "b": 1, "c": {"d": [1, 2], "e": "FILL"}}
    assert cm.matches_template(t, {"a": [3], "b": 1, "c": {"d": [1, 2], "e": {"z": 1}}}) == []
    assert cm.matches_template(t, {"a": 1, "b": 2, "c": {"d": [1, 2], "e": 0}}) == [".b: 1 -> 2"]
    assert cm.matches_template(t, {"a": 1, "b": 1.0, "c": {"d": [1, 2], "e": 0}}) == [".b: 1 -> 1.0"]
    assert cm.matches_template(t, {"a": 1, "b": 1, "c": {"d": [1], "e": 0}}) == [".c.d: list differs"]
    assert cm.matches_template(t, {"a": 1, "b": 1, "x": 0, "c": {"d": [1, 2], "e": 0}}) == [": keys differ"]


def test_filled_conditions_of_the_tests_keep_the_template(tmp_path):
    c1 = work1_conditions(tmp_path, 45)
    diff = cm.matches_template(json.loads((ROOT / "works/conditions/work1.template.json").read_text()), c1)
    assert diff == [".relief.L_mm: 50.0 -> 5.0", ".relief.eps: 0.3529079159926097 -> 3.529079159926097",
                    ".relief.grid_intervals: 250 -> 25"]  # the test tile is small on purpose
    c2 = work2_conditions(tmp_path)
    assert cm.matches_template(json.loads((ROOT / "works/conditions/work2.template.json").read_text()), c2) == []


def test_gates(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="do not exist"):
        cm.git_state(["src", "no/such/path"])
    assert cm.git_state(["src"])["git_commit"]
    with pytest.raises(SystemExit, match="not"):
        cm.check_versions({"fugashi": "0.0.0"}, ("fugashi",))
    cm.check_versions({"fugashi": "1.5.2", "ipadic": "1.0.0"}, ("fugashi", "ipadic"))
    g = json.loads((ROOT / "works/conditions/work1.template.json").read_text())["generator"]
    cm.check_generator(g)
    with pytest.raises(SystemExit, match="installed from"):
        cm.check_generator({**g, "commit": "0" * 40})
    with pytest.raises(SystemExit, match="library files"):
        cm.check_generator({**g, "library_sha256": {**g["library_sha256"], "field.py": "0" * 64}})


def test_same_as_head_on_a_temporary_repository(tmp_path, monkeypatch):
    import subprocess

    def g(*a):
        subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)
    g("init", "-q")
    (tmp_path / "a.txt").write_text("one\n")
    g("add", "a.txt")
    g("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "a")
    monkeypatch.setattr(cm, "ROOT", tmp_path.resolve())
    cm.same_as_head([tmp_path / "a.txt"])
    assert cm.git_state(["a.txt"])["git_dirty"] is False
    (tmp_path / "a.txt").write_text("two\n")
    with pytest.raises(SystemExit, match="differs"):
        cm.same_as_head([tmp_path / "a.txt"])
    assert cm.git_state(["a.txt"])["git_dirty"] is True
    (tmp_path / "b.txt").write_text("new\n")
    with pytest.raises(SystemExit, match="differs"):
        cm.same_as_head([tmp_path / "b.txt"])  # not committed


def test_work1_s_max_below_the_rule(tmp_path):
    c = work1_conditions(tmp_path, 45)
    c["relief"]["s_max"] = 0.25
    with pytest.raises(SystemExit, match="s_max"):
        work1.validate(c)


def test_stretch_rejects_non_finite_points():
    import stretch
    from window import WindowPoint

    with pytest.raises(ValueError):
        stretch.stretch([WindowPoint(0, 0.5, 40, 0.1, float("nan"))], 2.0)
    with pytest.raises(ValueError):
        stretch.reference_values([[WindowPoint(0, 0.5, 40, float("inf"), 0.0)]])
