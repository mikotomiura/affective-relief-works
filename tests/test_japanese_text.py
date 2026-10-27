"""Tests of the Japanese text step on sentences written for the tests (no text of any work, no real lexicon)."""

from __future__ import annotations

import pytest

from japanese_text import (GAIJI, POS_ALLOWED, Lexicon, LinearMap, Tagger, Token, is_candidate, load_memolon,
                           sentence_point, split_line, split_sentences, strip_aozora, text_path)

IDENTITY = LinearMap(1.0, 0.0, 1.0, 0.0)  # values of the fake lexicons below are already in [-1, 1] or test the clip

HEADER = "題\n著者\n\n" + "-" * 55 + "\n【テキスト中に現れる記号について】\n\n《》：ルビ\n" + "-" * 55 + "\n"
FOOTER = "\n\n底本：「本」出版社\n入力：誰か\n"


@pytest.fixture(scope="module")
def tagger() -> Tagger:
    return Tagger()


def surfaces(tagger: Tagger, s: str, select: str = "pos") -> list[str]:
    return [t.surface for t in tagger.tokens(s) if is_candidate(t, select)]


# ---------------------------------------------------------------- Aozora


def test_header_and_footer_are_removed():
    body, rep = strip_aozora(HEADER + "本文の一行目。\n二行目。" + FOOTER)
    assert body.split("\n")[0] == "本文の一行目。"
    assert "底本" not in body and "記号について" not in body and "著者" not in body
    assert rep.header_lines == 8


def test_end_of_text_note_is_a_footer_when_there_is_no_teihon():
    body, rep = strip_aozora(HEADER + "本文。\n［＃本文終わり］\n入力：誰か\n")
    assert body.strip() == "本文。" and rep.footer_mark == "本文終わり"


def test_the_earlier_of_the_two_footer_marks_ends_the_body():
    body, rep = strip_aozora(HEADER + "本文。最後の文。［＃本文終わり］後記。\n後記の文。" + FOOTER)
    assert body.strip() == "本文。最後の文。" and rep.footer_mark == "本文終わり"
    body, rep = strip_aozora(HEADER + "本文。" + FOOTER + "［＃本文終わり］\n")
    assert body.strip() == "本文。" and rep.footer_mark == "底本"


def test_missing_header_or_footer_fails():
    with pytest.raises(ValueError, match="header"):
        strip_aozora("題\n本文。" + FOOTER)
    with pytest.raises(ValueError, match="footer"):
        strip_aozora(HEADER + "本文。\n")


def test_crlf_is_accepted():
    body, _ = strip_aozora((HEADER + "本文。" + FOOTER).replace("\n", "\r\n"))
    assert body.strip() == "本文。"


def test_ruby_and_its_start_mark_are_removed():
    body, rep = strip_aozora(HEADER + "霧の｜ロンドン警視庁《スコットランドヤード》に青空《あおぞら》。" + FOOTER)
    assert body.strip() == "霧のロンドン警視庁に青空。"
    assert rep.ruby == 2


def test_nested_note_and_external_character():
    line = "［＃注記付き］勝安房守［＃「本ト麟太郎※［＃コト、1-2-24］」の注記付き終わり］が水※［＃「木＋靈」、第3水準1-86-29］子窓。"
    body, rep = strip_aozora(HEADER + line + FOOTER)
    assert body.strip() == f"勝安房守が水{GAIJI}子窓。"
    assert rep.gaiji == 1 and rep.notes == 2


def test_unclosed_note_fails():
    with pytest.raises(ValueError, match="unclosed"):
        strip_aozora(HEADER + "本文［＃傍点" + FOOTER)


def test_headings_of_three_forms_are_removed_with_their_text():
    text = (HEADER + "［＃２字下げ］上　先生と私［＃「上　先生と私」は大見出し］\n"
            "独《ひと》り寝《ね》の別《わか》れ［＃「独り寝の別れ」は中見出し］\n"
            "［＃中見出し］亜細亜《アジア》の曙《あけぼの》［＃中見出し終わり］\n"
            "見出し［＃「見出し」は同行中見出し］　ここは本文である。\n"
            "［＃ここから大見出し］\n一行目の見出し\n二行目の見出し\n［＃ここで大見出し終わり］\n"
            "最後の本文。" + FOOTER)
    body, rep = strip_aozora(text)
    lines = [ln for ln in body.split("\n") if ln]
    assert lines == ["ここは本文である。", "最後の本文。"]
    assert rep.headings == 5


def test_heading_note_that_does_not_follow_its_text_fails():
    with pytest.raises(ValueError, match="does not follow"):
        strip_aozora(HEADER + "別の文字［＃「見出し」は中見出し］" + FOOTER)


def test_unclosed_heading_block_fails():
    with pytest.raises(ValueError, match="not closed"):
        strip_aozora(HEADER + "［＃ここから中見出し］\n見出し\n" + FOOTER)


def test_accent_brackets_and_indentation():
    body, _ = strip_aozora(HEADER + "　繁雑な日本の 〔e'tiquette〕 も、〔亀甲〕の括弧。" + FOOTER)
    assert body.strip() == "繁雑な日本の e'tiquette も、〔亀甲〕の括弧。"


def test_heading_block_on_one_line_fails():
    with pytest.raises(ValueError, match="one line"):
        strip_aozora(HEADER + "［＃ここから中見出し］［＃ここで中見出し終わり］\n本文。" + FOOTER)


def test_note_that_only_mentions_the_word_heading_is_removed():
    body, _ = strip_aozora(HEADER + "見出しという語［＃「見出し」に傍点］。" + FOOTER)
    assert body.strip() == "見出しという語。"


def test_text_beside_a_heading_block_note_fails():
    with pytest.raises(ValueError, match="beside"):
        strip_aozora(HEADER + "本文［＃ここから中見出し］\n見出し\n［＃ここで中見出し終わり］\n" + FOOTER)
    with pytest.raises(ValueError, match="beside"):
        strip_aozora(HEADER + "［＃ここから中見出し］\n見出し\n［＃ここで中見出し終わり］本文\n" + FOOTER)


# ---------------------------------------------------------------- sentences


@pytest.mark.parametrize(("line", "want"), [
    ("晴れた。雨が降る！本当か？", ["晴れた。", "雨が降る！", "本当か？"]),
    ("何だって！？　そうか。", ["何だって！？", "そうか。"]),
    ("「行こう。」と彼は言った。", ["「行こう。」", "と彼は言った。"]),
    ("「行こう」と彼は言った。", ["「行こう」と彼は言った。"]),
    ("「行こう」「待って」", ["「行こう」", "「待って」"]),
    ("「行こう」　「待って」", ["「行こう」", "「待って」"]),
    ("「外の『内側』まで」と言った。", ["「外の『内側』まで」と言った。"]),
    ("「外の『内側』」「次」", ["「外の『内側』」", "「次」"]),
    ("（注のような括弧）のあと。", ["（注のような括弧）のあと。"]),
    ("（注）「次」", ["（注）「次」"]),
    ("（注）", ["（注）"]),
    ("句点のない行", ["句点のない行"]),
    ("＊　＊　＊", []),
    ("……。", []),
    (f"{GAIJI}。", []),
])
def test_split_line(line, want):
    assert split_line(line) == want


def test_line_breaks_end_sentences_and_empty_lines_vanish():
    kept, dropped = split_sentences("一行目の途中\n\n二行目。三文目。\n＊　＊　＊")
    assert kept == ["一行目の途中", "二行目。", "三文目。"] and dropped == ["＊　＊　＊"]


def test_cleaning_then_splitting():
    body, _ = strip_aozora(HEADER + "「甲」〔乙〕と言った。\n「行こう」［＃傍点］「待って」\n［＃改ページ］\n" + FOOTER)
    kept, _ = split_sentences(body)
    assert kept == ["「甲」", "〔乙〕と言った。", "「行こう」", "「待って」"]  # 〔 kept, so it opens a bracket


# ---------------------------------------------------------------- parts of speech


def test_allowed_parts_of_speech(tagger):
    got = surfaces(tagger, "私は昨日、東京の図書館で三冊の本を読んでしまった。")
    assert got == ["昨日", "図書館", "本", "読ん"]  # pronoun, proper noun, number, suffix, しまう are left out


def test_adjectival_noun_in_and_non_independent_out(tagger):
    assert surfaces(tagger, "それは静かなことだ。") == ["静か"]
    assert surfaces(tagger, "思い出しやすい季節だ。") == ["思い出し", "季節"]


def test_every_allowed_part_of_speech_appears(tagger):
    toks = tagger.tokens("美しい花をゆっくり見て、もっと遠くへ勉強しに行く。とんでもない話だ。")
    got = {(t.surface, t.pos1, t.pos2) for t in toks if is_candidate(t, "pos")}
    want = {("美しい", "形容詞", "自立"), ("花", "名詞", "一般"), ("ゆっくり", "副詞", "助詞類接続"),
            ("見", "動詞", "自立"), ("もっと", "副詞", "一般"), ("遠く", "名詞", "副詞可能"),
            ("勉強", "名詞", "サ変接続"), ("し", "動詞", "自立"), ("行く", "動詞", "自立"), ("とんでも", "名詞", "ナイ形容詞語幹"),
            ("話", "名詞", "サ変接続")}
    assert got == want
    assert {(p1, p2) for _, p1, p2 in got} == {(p1, p2) for p1, s in POS_ALLOWED.items() for p2 in s} - {
        ("名詞", "形容動詞語幹")}  # 形容動詞語幹 is covered by test_adjectival_noun_in_and_non_independent_out


def test_allowed_table_on_made_up_tokens():
    for p1, p2 in [("名詞", "固有名詞"), ("名詞", "代名詞"), ("名詞", "数"), ("名詞", "非自立"), ("名詞", "接尾"),
                   ("名詞", "特殊"), ("動詞", "非自立"), ("動詞", "接尾"), ("形容詞", "非自立"), ("形容詞", "接尾"),
                   ("助詞", "格助詞"), ("助動詞", "*"), ("連体詞", "*"), ("感動詞", "*"), ("記号", "一般")]:
        assert not is_candidate(Token("x", p1, p2), "pos"), (p1, p2)
    for p1, s in POS_ALLOWED.items():
        for p2 in s:
            assert is_candidate(Token("x", p1, p2), "pos")


def test_all_form_keeps_function_words_but_not_symbols(tagger):
    got = surfaces(tagger, "それは静かなことだ。", select="all")
    assert "は" in got and "こと" in got and "。" not in got


def test_unknown_selection_or_order_fails_even_when_nothing_matches(tagger):
    with pytest.raises(ValueError):
        sentence_point("本。", Lexicon({}), tagger, IDENTITY, select="nouns")
    with pytest.raises(ValueError):
        sentence_point("それは私だ。", Lexicon({}), tagger, IDENTITY, order="later")


# ---------------------------------------------------------------- points


def test_mean_of_matched_words_and_counts(tagger):
    lex = Lexicon({"静か": (0.4, -0.6), "季節": (0.0, 0.2), "は": (0.9, 0.9)})
    p = sentence_point("季節は静かだ。", lex, tagger, IDENTITY)
    assert p.matched == ["季節", "静か"] and p.Z == 2 and p.n_candidates == 2
    assert p.V == pytest.approx(0.2) and p.A == pytest.approx(-0.2)


def test_repeated_word_counts_each_time(tagger):
    lex = Lexicon({"本": (1.0, 0.0), "季節": (0.0, 0.0)})
    p = sentence_point("本と本と季節。", lex, tagger, IDENTITY)
    assert p.Z == 3 and p.V == pytest.approx(2 / 3)


def test_unmatched_sentence_is_neutral_and_flagged_but_kept_in_place(tagger):
    lex = Lexicon({"季節": (0.5, 0.5)})
    path = text_path(["季節。", "それは私だ。", "季節。"], lex, tagger, IDENTITY)
    assert [p.unmatched for p in path] == [False, True, False]
    assert (path[1].V, path[1].A, path[1].Z, path[1].n_candidates) == (0.0, 0.0, 0, 0)


def test_candidates_without_entries_are_counted(tagger):
    p = sentence_point("季節の図書館。", Lexicon({"季節": (0.1, 0.1)}), tagger, IDENTITY)
    assert (p.Z, p.n_candidates) == (1, 2)


def test_order_of_clip_and_mean(tagger):
    lex = Lexicon({"本": (3.0, 0.0), "季節": (-1.0, 0.0)})
    word = sentence_point("本と季節。", lex, tagger, IDENTITY, order="word")
    sent = sentence_point("本と季節。", lex, tagger, IDENTITY, order="sentence")
    assert word.V == pytest.approx(0.0)  # clip(3) = 1, then (1 + -1) / 2
    assert sent.V == pytest.approx(1.0)  # (3 + -1) / 2 = 1, then clip
    with pytest.raises(ValueError):
        sentence_point("本。", lex, tagger, IDENTITY, order="later")


def test_unknown_options_fail_on_an_empty_path(tagger):
    with pytest.raises(ValueError):
        text_path([], Lexicon({}), tagger, IDENTITY, select="nouns")
    with pytest.raises(ValueError):
        text_path([], Lexicon({}), tagger, IDENTITY, order="later")


def test_overflow_in_the_sentence_order_is_rejected(tagger):
    lex = Lexicon({"本": (1e308, 0.0)})
    with pytest.raises(ValueError):
        sentence_point("本と本。", lex, tagger, LinearMap(1e-308, -1.0, 1.0, 0.0), order="sentence")


def test_non_finite_values_are_rejected():
    with pytest.raises(ValueError):
        LinearMap(float("nan"), 0.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        LinearMap(1.0, float("inf"), 1.0, 0.0)
    with pytest.raises(ValueError):
        LinearMap(1.0, 0.0, 1.0, 0.0)(float("nan"), 0.0)


def test_linear_map_and_clip():
    m = LinearMap(0.5, -2.0, 1.0, 0.0)
    assert m(5.0, 0.3) == (0.5, 0.3)
    assert m(10.0, -4.0) == (1.0, -1.0)
    assert m.raw(10.0, -4.0) == (3.0, -4.0)


def test_no_unicode_normalisation(tagger):
    lex = Lexicon({"cafe": (0.5, 0.5)})  # half-width only; IPAdic tags both forms as nouns
    assert sentence_point("cafeの季節。", lex, tagger, IDENTITY, select="all").matched == ["cafe"]
    assert sentence_point("ｃａｆｅの季節。", lex, tagger, IDENTITY, select="all").matched == []


# ---------------------------------------------------------------- lexicon file


def test_load_memolon(tmp_path):
    p = tmp_path / "ja.tsv"
    p.write_text("word\tvalence\tarousal\tdominance\n季節\t5.5\t4.1\t5.0\n本\t6.0\t3.9\t5.2\n", encoding="utf-8")
    lex = load_memolon(p)
    assert lex.entries == {"季節": (5.5, 4.1), "本": (6.0, 3.9)} and len(lex.sha256) == 64


@pytest.mark.parametrize("body", ["季節\t5.5\t4.1\n季節\t5.0\t4.0\n", "季節\tnan\t4.1\n", "季節\t5.5\n", "\t5.5\t4.1\n",
                                  "季節 \t5.5\t4.1\n"])
def test_load_memolon_rejects_bad_files(tmp_path, body):
    p = tmp_path / "bad.tsv"
    p.write_text("word\tvalence\tarousal\n" + body, encoding="utf-8")
    with pytest.raises(ValueError):
        load_memolon(p)


def test_load_memolon_rejects_wrong_header(tmp_path):
    p = tmp_path / "bad.tsv"
    p.write_text("term\tvalence\tarousal\n季節\t5.5\t4.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        load_memolon(p)


def test_header_with_one_line_of_hyphens_fails():
    with pytest.raises(ValueError, match="header"):
        strip_aozora("題\n著者\n" + "-" * 55 + "\n本文。" + FOOTER)


def test_heading_note_outside_the_heading_rules_fails():
    # a forward heading note whose text holds an external character note is not taken by the heading rules
    with pytest.raises(ValueError, match="unhandled heading"):
        strip_aozora(HEADER + "本※［＃米印、1-2-8］［＃「本※［＃米印、1-2-8］」は中見出し］" + FOOTER)
