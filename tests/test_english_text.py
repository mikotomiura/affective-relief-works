"""Tests of the English text step on lines written for the tests (no text of any work, no lexicon)."""

from __future__ import annotations

import pytest

from english_text import extract_story, split_paragraph, split_sentences


def gutenberg(body: str, contents: str = "") -> str:
    return ("The Project Gutenberg eBook of Tests\r\n\r\n*** START OF THE PROJECT GUTENBERG EBOOK TESTS ***\r\n"
            + contents + body + "\r\n*** END OF THE PROJECT GUTENBERG EBOOK TESTS ***\r\nLicense text.\r\n")


BOOK = gutenberg(
    "\r\nFIRST STORY\r\n\r\n\r\nThe rain fell. It fell on the\r\nroof all night.\r\n\r\n[Picture: A roof]\r\n\r\n"
    "A second paragraph.\r\n\r\n\r\nSECOND STORY\r\n\r\nAnother story.\r\n",
    contents="CONTENTS\r\n\r\nFIRST STORY\r\nSECOND STORY\r\n")


# ---------------------------------------------------------------- extraction


def test_story_between_headings_with_joined_lines_and_dropped_illustration():
    paras, rep = extract_story(BOOK, "FIRST STORY", "SECOND STORY", occurrence=1)
    assert paras == ["The rain fell. It fell on the roof all night.", "A second paragraph."]
    assert rep.dropped_brackets == ["[Picture: A roof]"]
    assert rep.paragraphs == 2 and len(rep.body_sha256) == 64


def test_the_occurrence_picks_the_heading_line():
    # occurrence 0 is the line of the table of contents: the next line is the next heading, so the body is empty
    with pytest.raises(ValueError, match="no paragraph"):
        extract_story(BOOK, "FIRST STORY", "SECOND STORY", occurrence=0)
    with pytest.raises(ValueError, match="occurs 2 times"):
        extract_story(BOOK, "FIRST STORY", "SECOND STORY", occurrence=2)


def test_last_story_ends_at_the_end_marker():
    paras, _ = extract_story(BOOK, "SECOND STORY", None, occurrence=-1)
    assert paras == ["Another story."]


def test_licence_after_the_end_marker_is_not_read():
    paras, _ = extract_story(BOOK, "SECOND STORY", None, occurrence=-1)
    assert all("License" not in p for p in paras)


def test_missing_markers_or_headings_fail():
    with pytest.raises(ValueError, match="START and one END"):
        extract_story("FIRST STORY\n\nText.\n", "FIRST STORY", None)
    with pytest.raises(ValueError, match="not found"):
        extract_story(BOOK, "FIRST STORY", "THIRD STORY", occurrence=1)
    with pytest.raises(ValueError, match="occurs 0 times"):
        extract_story(BOOK, "NO STORY", None)


def test_a_bracket_inside_a_paragraph_is_kept():
    b = gutenberg("\nONE\n\nHe said [softly] nothing.\n")
    paras, rep = extract_story(b, "ONE", None)
    assert paras == ["He said [softly] nothing."] and rep.dropped_brackets == []


# ---------------------------------------------------------------- sentences


def test_plain_terminal_marks_split():
    assert split_paragraph("It rained. Was it cold? Yes! It was.") == ["It rained.", "Was it cold?", "Yes!", "It was."]


def test_a_closing_quote_after_the_terminal_mark_still_ends_the_sentence():
    assert split_paragraph("“Go home.” He went. She said, “Now?” “Yes.”") == [
        "“Go home.”", "He went.", "She said, “Now?”", "“Yes.”"]


def test_a_lower_case_word_after_the_mark_does_not_start_a_sentence():
    assert split_paragraph("“Where is it?” she asked. Tum-tum! tiddle-um!") == [
        "“Where is it?” she asked.", "Tum-tum! tiddle-um!"]


def test_title_abbreviations_do_not_end_a_sentence():
    assert split_paragraph("Mrs. Sommers went out. She met Dr. Brown and Mr. Smith.") == [
        "Mrs. Sommers went out.", "She met Dr. Brown and Mr. Smith."]


def test_an_abbreviation_must_be_a_whole_word():
    # "Hamr" is not "Mr"; the stop after it ends the sentence
    assert split_paragraph("He met Hamr. Then he left.") == ["He met Hamr.", "Then he left."]
    # the regular expression reads ASCII letters only, so a word ending in "Mr" after a non-ASCII letter is checked too
    assert split_paragraph("He met ÉMr. Then he left.") == ["He met ÉMr.", "Then he left."]


def test_other_abbreviations_are_not_protected():
    assert split_paragraph("He met Col. Brown.") == ["He met Col.", "Brown."]


def test_ellipsis_and_runs_of_marks():
    assert split_paragraph("Again!... But no. Wait.... Little one?! Yes…  Gone.") == [
        "Again!...", "But no.", "Wait....", "Little one?!", "Yes…", "Gone."]


def test_dashes_and_marks_without_space_do_not_split():
    assert split_paragraph("It was red—the redness of blood. A.B.C. was there.") == [
        "It was red—the redness of blood.", "A.B.C. was there."]


def test_digits_and_opening_brackets_start_a_sentence():
    assert split_paragraph("He waited. 1842 came. (It was late.) End.") == [
        "He waited.", "1842 came.", "(It was late.)", "End."]


def test_unicode_upper_case_starts_a_sentence():
    assert split_paragraph("He stopped. Émile came.") == ["He stopped.", "Émile came."]


def test_paragraph_end_always_ends_a_sentence():
    kept, dropped = split_sentences(["He said nothing", "and she left."])
    assert kept == ["He said nothing", "and she left."] and dropped == []


def test_pieces_without_letters_are_not_sentences():
    kept, dropped = split_sentences(["Text.", "* * *", "More."])
    assert kept == ["Text.", "More."] and dropped == ["* * *"]


def test_heading_lines_with_surrounding_space_are_matched_after_stripping():
    b = gutenberg("\n   ONE  \n\nText one.\n\n\t TWO\n\nText two.\n")
    assert extract_story(b, "ONE", "TWO")[0] == ["Text one."]


def test_lone_carriage_returns_are_line_breaks():
    b = gutenberg("\rONE\r\rText.\r")
    assert extract_story(b, "ONE", None)[0] == ["Text."]


def test_negative_occurrence_out_of_range_fails():
    with pytest.raises(ValueError, match="occurs 2 times"):
        extract_story(BOOK, "FIRST STORY", "SECOND STORY", occurrence=-3)


def test_nested_brackets_are_a_bracketed_paragraph_but_two_notes_are_not():
    b = gutenberg("\nONE\n\n[Picture: A [small] roof]\n\n[Note] real text [Note]\n")
    paras, rep = extract_story(b, "ONE", None)
    assert rep.dropped_brackets == ["[Picture: A [small] roof]"] and paras == ["[Note] real text [Note]"]


@pytest.mark.parametrize("abbr", ["Mr", "Mrs", "Dr", "St", "Messrs", "Mme", "Mlle"])
def test_every_listed_abbreviation_is_protected(abbr):
    assert split_paragraph(f"They saw {abbr}. Brown there. Then left.") == [f"They saw {abbr}. Brown there.", "Then left."]


def test_only_a_single_full_stop_after_an_abbreviation_is_protected():
    assert split_paragraph("He met Mr.. Then left. Was it Mr? Yes.") == ["He met Mr..", "Then left.", "Was it Mr?", "Yes."]


def test_abbreviation_boundary_with_upper_case_letters_and_digits():
    # "AMr" and "2Mrs" are not the listed words, so the stop after them ends the sentence
    assert split_paragraph("He met AMr. Then left.") == ["He met AMr.", "Then left."]
    assert split_paragraph("Meet 2Mrs. Next.") == ["Meet 2Mrs.", "Next."]


@pytest.mark.parametrize("opener", ["[", "‘", "“", "\"", "'", "’"])
def test_opening_marks_start_a_sentence(opener):
    assert split_paragraph(f"He left. {opener}Twas night.") == ["He left.", f"{opener}Twas night."]


def test_no_break_space_counts_as_white_space():
    assert split_paragraph("He went.\u00a0Then he came.") == ["He went.", "Then he came."]


def test_pieces_without_letters_are_returned_by_split_paragraph():
    assert split_paragraph("* * *") == ["* * *"]
