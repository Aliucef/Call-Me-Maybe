"""Unit tests for string_decoder — LLM is mocked."""
import pytest

from src.decoder.string_decoder import (
    choose_string,
    extract_quoted_spans,
    extract_regex_pattern_candidates,
    extract_trailing_word_candidate,
    resolve_string,
)
from tests.conftest import MockLLM


class TestExtractQuotedSpans:
    def test_mixed_quote_styles_in_order(self) -> None:
        prompt = "Substitute the word 'cat' with \"dog\" in 'the cat sat'"
        assert extract_quoted_spans(prompt) == ["cat", "dog", "the cat sat"]

    def test_no_quotes_returns_empty(self) -> None:
        assert extract_quoted_spans("Greet john") == []


class TestExtractRegexPatternCandidates:
    def test_recognizes_known_concept_word(self) -> None:
        assert extract_regex_pattern_candidates("Replace all vowels here") == [r"[aeiouAEIOU]"]

    def test_deduplicates_same_pattern(self) -> None:
        result = extract_regex_pattern_candidates("Replace all numbers, every number")
        assert result == [r"\d+"]

    def test_unknown_words_yield_no_candidates(self) -> None:
        assert extract_regex_pattern_candidates("Greet john") == []


class TestExtractTrailingWordCandidate:
    def test_returns_last_word_when_enough_words(self) -> None:
        assert extract_trailing_word_candidate("Greet john") == ["john"]

    def test_single_word_returns_empty(self) -> None:
        assert extract_trailing_word_candidate("Greet") == []


class TestChooseString:
    def test_prefers_quoted_span_over_regex_concept(self) -> None:
        llm = MockLLM()
        llm.prefer("hello")
        result = choose_string(llm, "Reverse the vowels in 'hello'", param_name="s")
        assert result == "hello"

    def test_falls_back_to_regex_concept_when_unquoted(self) -> None:
        llm = MockLLM()
        llm.prefer(r"[aeiouAEIOU]")
        result = choose_string(llm, "Replace all vowels with asterisks", param_name="regex")
        assert result == r"[aeiouAEIOU]"

    def test_falls_back_to_trailing_word_as_last_resort(self) -> None:
        llm = MockLLM()
        llm.prefer("john")
        result = choose_string(llm, "Greet john", param_name="name")
        assert result == "john"

    def test_returns_none_when_nothing_extractable(self) -> None:
        llm = MockLLM()
        assert choose_string(llm, "Greet", param_name="name") is None

    def test_excludes_already_used_values(self) -> None:
        llm = MockLLM()
        llm.prefer("dog")
        result = choose_string(
            llm,
            "Substitute 'cat' with 'dog' in 'the cat sat'",
            param_name="replacement",
            already_extracted={"source_string": "the cat sat", "regex": "cat"},
        )
        assert result == "dog"


class TestResolveString:
    def test_returns_choice_when_available(self) -> None:
        llm = MockLLM()
        llm.prefer("john")
        assert resolve_string(llm, "Greet john", param_name="name") == "john"

    def test_defaults_to_empty_string_when_absent(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        llm = MockLLM()
        result = resolve_string(llm, "Greet", param_name="name")
        assert result == ""
        assert "[ERROR]" in capsys.readouterr().out
