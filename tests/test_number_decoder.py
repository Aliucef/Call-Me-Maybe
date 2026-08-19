"""Unit tests for number_decoder — LLM is mocked."""
import pytest

import src.decoder.number_decoder as number_decoder
from src.decoder.number_decoder import (
    choose_number,
    encode_number_candidate,
    extract_number_candidates,
    resolve_number,
)
from tests.conftest import MockLLM


@pytest.fixture(autouse=True)
def _clear_encode_cache() -> None:
    """Reset the module-level number-encoding cache before each test."""
    number_decoder._number_encode_cache.clear()


class TestExtractNumberCandidates:
    def test_finds_all_literals_in_order(self) -> None:
        assert extract_number_candidates("sum of 2 and 3.5, then -4") == ["2", "3.5", "-4"]

    def test_no_numbers_returns_empty(self) -> None:
        assert extract_number_candidates("Greet john") == []


class TestChooseNumber:
    def test_picks_the_preferred_candidate(self) -> None:
        llm = MockLLM()
        llm.prefer("2")
        result = choose_number(llm, "What is the sum of 2 and 3?", param_name="a")
        assert result == 2.0

    def test_no_numbers_returns_none(self) -> None:
        llm = MockLLM()
        assert choose_number(llm, "Greet john") is None

    def test_occurrence_counting_allows_repeated_value(self) -> None:
        """"sum of 2 and 2" has two real occurrences of 2, so the second
        parameter must still resolve to 2.0 instead of running out."""
        llm = MockLLM()
        llm.prefer("2")
        first = choose_number(llm, "sum of 2 and 2", param_name="a")
        assert first == 2.0
        llm.prefer("2")
        second = choose_number(
            llm, "sum of 2 and 2", param_name="b", already_extracted={"a": first},
        )
        assert second == 2.0

    def test_occurrence_counting_exhausts_single_value(self) -> None:
        """"sum of 2 and ?" has only one occurrence of 2, so the second
        parameter must NOT silently reuse it."""
        llm = MockLLM()
        llm.prefer("2")
        first = choose_number(llm, "sum of 2 and ?", param_name="a")
        assert first == 2.0
        second = choose_number(
            llm, "sum of 2 and ?", param_name="b", already_extracted={"a": first},
        )
        assert second is None


class TestResolveNumber:
    def test_returns_choice_when_available(self) -> None:
        llm = MockLLM()
        llm.prefer("16")
        assert resolve_number(llm, "square root of 16", param_name="a") == 16.0

    def test_defaults_to_zero_when_absent(self, capsys: pytest.CaptureFixture[str]) -> None:
        llm = MockLLM()
        result = resolve_number(llm, "Greet john", param_name="a")
        assert result == 0.0
        assert "[ERROR]" in capsys.readouterr().out


class TestEncodeNumberCandidateCache:
    def test_second_call_hits_cache(self) -> None:
        llm = MockLLM()
        first_ids = encode_number_candidate(llm, "42")
        assert llm.encode_calls == ["42"]
        second_ids = encode_number_candidate(llm, "42")
        assert llm.encode_calls == ["42"]
        assert first_ids == second_ids

    def test_different_literals_both_encode(self) -> None:
        llm = MockLLM()
        encode_number_candidate(llm, "1")
        encode_number_candidate(llm, "2")
        assert llm.encode_calls == ["1", "2"]

    def test_shared_literal_across_prompts_reuses_cache(self) -> None:
        """Simulates two prompts in the same run that both mention "2" --
        the second prompt's resolution should not re-encode it."""
        llm = MockLLM()
        llm.prefer("2")
        choose_number(llm, "sum of 2 and 3", param_name="a")
        assert llm.encode_calls.count("2") == 1
        llm.prefer("2")
        choose_number(llm, "sum of 2 and 5", param_name="a")
        assert llm.encode_calls.count("2") == 1
