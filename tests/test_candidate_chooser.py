"""Unit tests for candidate_chooser — the shared constrained-decoding primitive."""
import pytest

from src.decoder.candidate_chooser import choose_from_candidates
from tests.conftest import MockLLM


class TestChooseFromCandidates:
    def test_narrows_to_the_preferred_multi_token_candidate(self) -> None:
        candidates = {"cat": [1, 2], "dog": [3, 4]}
        llm = MockLLM(vocab_size=10, preferred_token_per_step=[1, 2])
        assert choose_from_candidates(llm, "ctx", candidates) == "cat"

    def test_verbose_does_not_raise(self) -> None:
        candidates = {"cat": [1, 2], "dog": [3, 4]}
        llm = MockLLM(vocab_size=10, preferred_token_per_step=[3, 4])
        assert choose_from_candidates(llm, "ctx", candidates, verbose=True) == "dog"

    def test_shared_prefix_resolves_to_the_longer_candidate(self) -> None:
        """Known behaviour, not a design goal: when one candidate's tokens
        are an exact prefix of another's ("cat" vs "cats"), the loop can
        only stop once a FULL candidate is matched, so it keeps extending
        past the shorter one -- "cat" ends up silently dropped in favour of
        "cats" even though it was already a complete, valid match. Pinned
        here so a future change to the narrowing logic doesn't alter this
        silently."""
        candidates = {"cat": [1, 2], "cats": [1, 2, 3]}
        llm = MockLLM(vocab_size=10, preferred_token_per_step=[1, 2, 3])
        assert choose_from_candidates(llm, "ctx", candidates) == "cats"

    def test_fully_tied_candidates_return_one_without_crashing(self) -> None:
        candidates = {"a": [5, 6], "b": [5, 6]}
        llm = MockLLM(vocab_size=10)
        assert choose_from_candidates(llm, "ctx", candidates) in ("a", "b")

    def test_empty_candidates_raises(self) -> None:
        """Known limitation: every current caller (function_chooser,
        number_decoder, string_decoder, boolean_decoder) already guards
        against an empty candidate set before calling in, so this path
        isn't reachable from the real pipeline today -- but calling this
        shared primitive directly with no candidates raises instead of
        degrading gracefully. Documented so it isn't rediscovered blind."""
        llm = MockLLM()
        with pytest.raises(ValueError):
            choose_from_candidates(llm, "ctx", {})
