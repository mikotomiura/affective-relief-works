"""The draw of the reference works on a fake list and a fake fetch (no network, no real text, a fake lexicon)."""

import io
import json
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "checks"))

import japanese_text  # noqa: E402
import reference_stretch as rs  # noqa: E402
from pipeline import calibration_map  # noqa: E402

COND = json.loads((ROOT / "checks/conditions/reference_stretch.json").read_text())
EL = COND["eligible"]


def row(wid, pid, role="著者", kana="新字新仮名", flag="なし", ndc="NDC 913", enc="ShiftJIS", name="姓"):
    return {"作品ID": wid, "人物ID": pid, "役割フラグ": role, "文字遣い種別": kana, "作品著作権フラグ": flag,
            "分類番号": ndc, "テキストファイル符号化方式": enc, "テキストファイルURL": f"https://x/{wid}.zip",
            "姓": name, "名": pid, "作品名": f"作品{wid}"}


def test_eligible():
    rows = [row("1", "p1"), row("2", "p1"), row("3", "p2"),
            row("4", "p3"), row("4", "000035", role="校訂者"),  # an excluded person in another role
            row("5", "p4"), row("5", "p5"),  # two authors
            row("6", "000148"),  # excluded author
            row("7", "p6", kana="旧字旧仮名"), row("8", "p6", flag="あり"), row("9", "p6", ndc="NDC 913 914"),
            row("10", "p6", enc="UTF-8"), row("11", "p6", ndc=" NDC 913 "), row("12", "p7", role="翻訳者")]
    got = rs.eligible(rows, EL)
    assert {p: [r["作品ID"] for r in ws] for p, ws in got.items()} == {"p1": ["1", "2"], "p2": ["3"], "p6": ["11"]}


def aozora(n: int) -> bytes:
    body = "\n".join(f"嬉しい朝に私は道を歩いた{i}。" if i % 4 else "ああ。" for i in range(n))
    text = "題\n作者\n\n" + "-" * 55 + "\n記号\n" + "-" * 55 + "\n" + body + "\n\n底本：本\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.txt", text.encode("cp932"))
    return buf.getvalue()


def fake_list(n_persons=8):
    rows = []
    for p in range(n_persons):
        for w in range(3):
            rows.append(row(f"{p}{w}", f"p{p}"))
    return rows


def fetcher(lengths):
    calls = []

    def fetch(url):
        calls.append(url)
        wid = url.rsplit("/", 1)[1][:-4]
        if lengths.get(wid) == "broken":
            return b"not a zip"
        return aozora(lengths.get(wid, 250))
    return fetch, calls


def test_draw_is_fixed_by_the_seed_and_skips(tmp_path):
    works = rs.eligible(fake_list(), EL)
    lengths = {f"{p}{w}": (100 if w == 0 else "broken" if w == 1 else 250) for p in range(8) for w in range(3)}
    dr = {"seed": 7, "n_works": 5}
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a, sa = rs.draw(works, dr, [200, 1000], tmp_path / "a", fetcher(lengths)[0])
    b, _ = rs.draw(works, dr, [200, 1000], tmp_path / "b", fetcher(lengths)[0])
    assert [w["work_id"] for w in a] == [w["work_id"] for w in b]
    assert len(a) == 5 and len({w["person"] for w in a}) == 5
    assert all(w["work_id"].endswith("2") for w in a)  # the short and the broken works are skipped
    assert {s["reason"].split(":")[0] for s in sa} <= {"100 sentences", "BadZipFile"}
    assert any(s["reason"] == "100 sentences" for s in sa) and any(s["reason"].startswith("BadZipFile") for s in sa)
    c, _ = rs.draw(works, {"seed": 8, "n_works": 5}, [200, 1000], tmp_path / "a", fetcher(lengths)[0])
    assert [w["work_id"] for w in c] != [w["work_id"] for w in a]  # another seed, another draw


def test_draw_uses_the_kept_zip_and_stops_when_persons_run_out(tmp_path):
    works = rs.eligible(fake_list(3), EL)
    fetch, calls = fetcher({})
    chosen, _ = rs.draw(works, {"seed": 1, "n_works": 3}, [200, 1000], tmp_path, fetch)
    n = len(calls)
    chosen2, _ = rs.draw(works, {"seed": 1, "n_works": 3}, [200, 1000], tmp_path, fetch)
    assert len(calls) == n and [w["zip_sha256"] for w in chosen] == [w["zip_sha256"] for w in chosen2]
    with pytest.raises(SystemExit, match="only 3"):
        rs.draw(works, {"seed": 1, "n_works": 4}, [200, 1000], tmp_path, fetch)


def test_summary_gives_k_from_the_windows(tmp_path):
    works = rs.eligible(fake_list(4), EL)
    chosen, _ = rs.draw(works, {"seed": 3, "n_works": 4}, [200, 1000], tmp_path, fetcher({})[0])
    lex = japanese_text.Lexicon({"嬉しい": (7.0, 5.0), "道": (5.0, 4.5)})
    vmap = calibration_map((ROOT / "results/calibrate_memolon.json").read_bytes())
    res = rs.summary(COND, chosen, lex, japanese_text.Tagger(), vmap)
    r = res["result"]
    assert r["k"] == pytest.approx(0.9 / r["Q"])
    assert 0.0 <= r["clip_rate"] <= 0.01 + 1e-12
    assert all("text" not in w for w in res["works"]) and len(res["works"]) == 4
    assert r["pooled_percentiles"]["stretched"]["V"]["50"] == pytest.approx(
        min(1, r["k"] * r["pooled_percentiles"]["unstretched"]["V"]["50"]))


def test_network_failures_stop_the_run(monkeypatch):
    def boom(*a, **k):
        raise OSError("timed out")
    monkeypatch.setattr(rs.time, "sleep", lambda s: None)
    monkeypatch.setattr(rs.urllib.request, "urlopen", boom)
    with pytest.raises(SystemExit, match="failed"):
        rs.fetch_url("https://x/1.zip")

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(rs.urllib.request, "urlopen", lambda *a, **k: Resp(b"<html>error</html>"))
    with pytest.raises(SystemExit, match="did not return a zip"):
        rs.fetch_url("https://x/1.zip")
    monkeypatch.setattr(rs.urllib.request, "urlopen", lambda *a, **k: Resp(aozora(10)))
    assert rs.fetch_url("https://x/1.zip") == aozora(10)


def test_a_failed_fetch_in_the_draw_is_not_a_skip(tmp_path):
    works = rs.eligible(fake_list(3), EL)

    def fetch(url):
        raise SystemExit("fetching failed")
    with pytest.raises(SystemExit, match="fetching failed"):
        rs.draw(works, {"seed": 1, "n_works": 2}, [200, 1000], tmp_path, fetch)
    assert not list(tmp_path.rglob("*.zip"))


def test_zips_are_kept_per_person(tmp_path):
    works = rs.eligible(fake_list(3), EL)
    chosen, _ = rs.draw(works, {"seed": 1, "n_works": 3}, [200, 1000], tmp_path, fetcher({})[0])
    for w in chosen:
        assert (tmp_path / w["person"] / f"{w['work_id']}.zip").exists()


@pytest.mark.parametrize("change", [
    lambda c: c["window"].update(W=30), lambda c: c["step"].update(select="all"),
    lambda c: c["k_rule"].update(quantile=0.95), lambda c: c["k_rule"].update(target=1.0),
    lambda c: c["draw"].update(n_works=1), lambda c: c["draw"].update(seed=-1), lambda c: c["eligible"].update(sentences=[200.0, 1000]),
])
def test_conditions_must_match_the_code(change):
    c = json.loads(json.dumps(COND))
    rs.validate(c)
    change(c)
    with pytest.raises(SystemExit):
        rs.validate(c)


def test_exclusion_is_checked_again_on_the_rows():
    rows = [row("1", "p1"), row("2", "p2"), row("2", "000119", role="編者")]
    chosen = [{"work_id": "1"}, {"work_id": "2"}]
    assert rs.excluded_in_rows(rows, chosen, set(EL["excluded_persons"])) == ["2"]
