"""Tests for the reverse Wordle solver.

Parsing and hard-mode tests are pure logic and run instantly. Reconstruction
tests need the precomputed pattern matrix (data/wordle/pattern_matrix.npy);
the first run generates it, which takes a few minutes.
"""

import pytest

from src.pattern import get_pattern
from src.reverse_solver import (
    merge_hard_mode_constraints,
    parse_pattern,
    parse_pattern_trace,
    parse_share_text,
    reconstruct_guesses,
    satisfies_hard_mode,
)

# ---------------------------------------------------------------------------
# parse_pattern
# ---------------------------------------------------------------------------


class TestParsePattern:
    def test_int_passthrough(self):
        assert parse_pattern(0) == 0
        assert parse_pattern(242) == 242

    def test_int_out_of_range(self):
        with pytest.raises(ValueError):
            parse_pattern(243)
        with pytest.raises(ValueError):
            parse_pattern(-1)

    def test_list(self):
        assert parse_pattern([1, 1, 0, 0, 0]) == 4
        assert parse_pattern((2, 2, 2, 2, 2)) == 242

    def test_list_bad_values(self):
        with pytest.raises(ValueError):
            parse_pattern([1, 1, 0, 0, 3])
        with pytest.raises(ValueError):
            parse_pattern([1, 1, 0, 0])

    def test_emoji(self):
        assert parse_pattern("🟨🟨⬛⬛⬛") == 4
        assert parse_pattern("🟩🟩🟩🟩🟩") == 242

    def test_white_square_is_gray(self):
        assert parse_pattern("⬜⬜⬜⬜⬜") == 0
        assert parse_pattern("🟨🟨⬜⬛⬜") == parse_pattern("🟨🟨⬛⬛⬛")

    def test_variation_selector_stripped(self):
        assert parse_pattern("🟨🟨⬛️⬛⬛") == 4

    def test_bad_emoji_string(self):
        with pytest.raises(ValueError):
            parse_pattern("🟨🟨⬛⬛")  # 4 tiles
        with pytest.raises(ValueError):
            parse_pattern("🟨🟨⬛⬛x")


# ---------------------------------------------------------------------------
# parse_pattern_trace
# ---------------------------------------------------------------------------


class TestParsePatternTrace:
    def test_valid(self):
        assert parse_pattern_trace([4, 109, 242]) == [4, 109, 242]

    def test_empty(self):
        with pytest.raises(ValueError):
            parse_pattern_trace([])

    def test_short_trace_must_end_green(self):
        with pytest.raises(ValueError, match="all-green"):
            parse_pattern_trace([4, 109])

    def test_intermediate_green_rejected(self):
        with pytest.raises(ValueError, match="ended"):
            parse_pattern_trace([242, 242])

    def test_lost_game_six_rows_accepted(self):
        trace = [4, 109, 144, 13, 40, 121]
        assert parse_pattern_trace(trace) == trace

    def test_lost_game_wrong_length_rejected(self):
        with pytest.raises(ValueError, match="lost game"):
            parse_pattern_trace([4, 109, 144, 13, 40])


# ---------------------------------------------------------------------------
# parse_share_text
# ---------------------------------------------------------------------------


class TestParseShareText:
    def test_full_share(self):
        text = "Wordle 1,784 4/6\n\n🟨🟨⬛⬛⬛\n🟨⬛⬛🟨🟨\n⬛⬛🟨🟩🟨\n🟩🟩🟩🟩🟩"
        assert parse_share_text(text) == [4, 109, 144, 242]

    def test_decorations_tolerated(self):
        # variation selector inside a row, emoji appended after a row,
        # emoji in the header
        text = "Wordle 999 3/6 😂\n\n⬜⬜⬜⬜⬜\n🟨⬛️⬛🟨🟨 😂\n🟩🟩🟩🟩🟩"
        assert parse_share_text(text) == [0, 109, 242]

    def test_malformed_row_raises_instead_of_dropping(self):
        text = "🟨🟨⬛⬛⬛\n⬛⬛🟨🟩\n🟩🟩🟩🟩🟩"
        with pytest.raises(ValueError, match="4 tiles"):
            parse_share_text(text)

    def test_no_rows(self):
        with pytest.raises(ValueError, match="No emoji rows"):
            parse_share_text("Wordle 999 X/6")

    def test_too_many_rows(self):
        text = "\n".join(["⬛⬛⬛⬛⬛"] * 7 + ["🟩🟩🟩🟩🟩"])
        with pytest.raises(ValueError, match="at most 6"):
            parse_share_text(text)

    def test_last_row_not_green(self):
        with pytest.raises(ValueError, match="all-green"):
            parse_share_text("🟨🟨⬛⬛⬛\n⬛⬛🟨🟩🟨")

    def test_lost_game_share(self):
        # "Wordle X/6" share: six rows, none all-green
        text = "Wordle 1,790 X/6\n\n" + "\n".join(
            ["🟨🟨⬛⬛⬛", "🟨⬛⬛🟨🟨", "⬛⬛🟨🟩🟨", "🟨⬛🟨⬛⬛", "⬛🟨⬛🟨⬛", "🟨🟩⬛⬛🟨"]
        )
        rows = parse_share_text(text)
        assert len(rows) == 6
        assert all(r != 242 for r in rows)


# ---------------------------------------------------------------------------
# Hard mode
# ---------------------------------------------------------------------------


class TestHardMode:
    def test_no_history_no_constraints(self):
        fixed, min_counts = merge_hard_mode_constraints([], [])
        assert fixed == {} and min_counts == {}
        assert satisfies_hard_mode("crane", fixed, min_counts)

    def test_green_fixes_position(self):
        # "store" with S green, everything else gray: pattern [2,0,0,0,0] = 2
        fixed, min_counts = merge_hard_mode_constraints(["store"], [2])
        assert fixed == {0: "s"}
        assert satisfies_hard_mode("slump", fixed, min_counts)
        assert not satisfies_hard_mode("crane", fixed, min_counts)

    def test_yellow_requires_letter_anywhere(self):
        # "raise" with R and A yellow: pattern [1,1,0,0,0] = 4
        fixed, min_counts = merge_hard_mode_constraints(["raise"], [4])
        assert fixed == {}
        assert min_counts == {"r": 1, "a": 1}
        assert satisfies_hard_mode("molar", fixed, min_counts)
        # gray letters MAY be reused in real hard mode
        assert satisfies_hard_mode("arise", fixed, min_counts)
        assert not satisfies_hard_mode("cloth", fixed, min_counts)

    def test_duplicate_letters_counted(self):
        # "geese" against an answer with two Es: greens+yellows of the same
        # letter require that many copies
        # pattern [0,2,1,0,0]: E green at pos 1, E yellow at pos 2
        pattern = 0 + 2 * 3 + 1 * 9
        fixed, min_counts = merge_hard_mode_constraints(["geese"], [pattern])
        assert fixed == {1: "e"}
        assert min_counts == {"e": 2}
        assert not satisfies_hard_mode("neath", fixed, min_counts)  # only one e
        assert satisfies_hard_mode("reede", fixed, min_counts)  # e in pos 1, two e's

    def test_constraints_merge_across_guesses(self):
        fixed, min_counts = merge_hard_mode_constraints(["raise", "molar"], [4, 13])
        # raise: r,a yellow; molar pattern 13 = [1,1,1,0,0]: m,o,l yellow
        assert min_counts["r"] == 1 and min_counts["a"] == 1
        assert min_counts["m"] == 1 and min_counts["o"] == 1 and min_counts["l"] == 1


# ---------------------------------------------------------------------------
# Reconstruction (requires pattern matrix)
# ---------------------------------------------------------------------------


def _assert_consistent(result: dict, patterns: list[int], answer: str):
    """Every reconstructed guess must produce the observed pattern."""
    for guess, pattern in zip(result["guesses"], patterns):
        assert int(get_pattern(guess, answer, "wordle")) == pattern, (
            f"{guess} vs {answer} does not produce pattern {pattern}"
        )


class TestReconstruction:
    PATTERNS = [4, 109, 144, 242]
    ANSWER = "umbra"

    def test_golden_umbra(self):
        results = reconstruct_guesses(patterns=self.PATTERNS, answer=self.ANSWER)
        assert results, "no reconstructions returned"
        top = results[0]
        assert top["guesses"][-1] == self.ANSWER
        _assert_consistent(top, self.PATTERNS, self.ANSWER)
        # sanity on the output schema
        assert len(top["step_info"]) == len(self.PATTERNS)
        assert 0 < top["normalized_probability"] <= 1

    def test_results_sorted_by_probability(self):
        results = reconstruct_guesses(patterns=self.PATTERNS, answer=self.ANSWER)
        log_probs = [r["log_probability"] for r in results]
        assert log_probs == sorted(log_probs, reverse=True)

    def test_out_of_list_answer_supported(self):
        # "soare" is a legal guess but not an official answer word. This trace
        # narrows the answer-list possibilities to zero, which used to collapse
        # the beam even though the trace is perfectly consistent.
        results = reconstruct_guesses(patterns=[202, 141, 242], answer="soare")
        assert results[0]["guesses"][-1] == "soare"
        _assert_consistent(results[0], [202, 141, 242], "soare")

    def test_intermediate_green_rejected(self):
        with pytest.raises(ValueError, match="ended"):
            reconstruct_guesses(patterns=[242, 242], answer="umbra")

    def test_hard_mode_reconstruction_honors_hints(self):
        results = reconstruct_guesses(
            patterns=self.PATTERNS, answer=self.ANSWER, hard_mode=True
        )
        for result in results[:10]:
            guesses = result["guesses"]
            _assert_consistent(result, self.PATTERNS, self.ANSWER)
            for k in range(1, len(guesses)):
                fixed, min_counts = merge_hard_mode_constraints(
                    guesses[:k], self.PATTERNS[:k]
                )
                assert satisfies_hard_mode(guesses[k], fixed, min_counts), (
                    f"{guesses[k]} violates hard mode after {guesses[:k]}"
                )

    def test_out_of_list_guesses_discounted(self):
        # With the discount disabled, obscure words may win; with the default
        # 10x discount, the reconstruction should favor answer-list words when
        # both match. The UMBRA trace has both kinds of candidates at step 2.
        results = reconstruct_guesses(patterns=self.PATTERNS, answer=self.ANSWER)
        from src.prior import get_word_list

        answer_set = set(get_word_list("wordle", short=True))
        top_words = results[0]["guesses"][:-1]
        assert all(w in answer_set for w in top_words), (
            f"expected answer-list words to win, got {top_words}"
        )

    def test_won_flag_set(self):
        results = reconstruct_guesses(patterns=self.PATTERNS, answer=self.ANSWER)
        assert results[0]["won"] is True

    def test_lost_game_reconstruction(self):
        # Build a genuine losing trace: six real guesses, none of them the answer
        answer = "vivid"
        guesses = ["crane", "moist", "gaudy", "pluck", "fewer", "begin"]
        patterns = [int(get_pattern(g, answer, "wordle")) for g in guesses]
        assert all(p != 242 for p in patterns)

        results = reconstruct_guesses(patterns=patterns, answer=answer)
        top = results[0]
        assert top["won"] is False
        assert len(top["guesses"]) == 6
        assert answer not in top["guesses"]
        # a plausible player never repeats a guess, even across identical rows
        assert len(set(top["guesses"])) == 6, f"repeated guess in {top['guesses']}"
        _assert_consistent(top, patterns, answer)

    def test_invalid_discount_rejected(self):
        with pytest.raises(ValueError, match="discount"):
            reconstruct_guesses(
                patterns=self.PATTERNS, answer=self.ANSWER, out_of_list_discount=0
            )
