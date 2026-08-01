"""Unit tests for number_decoder — no LLM required."""
from typing import Any

from src.decoder.number_decoder import (
    allowed_next_number_tokens,
    mask_logits_in_place,
    decode_number,
    _allowed_number_cache,
    NEG_INF,
)


# ---------------------------------------------------------------------------
# Minimal fake vocab for tests
# ---------------------------------------------------------------------------

VOCAB = ["", "1", "2", "3", ".", "-", "0", "abc", "10", " "]


def _ids(tokens: list[str]) -> set[int]:
    return {VOCAB.index(t) for t in tokens}


# ---------------------------------------------------------------------------
# allowed_next_number_tokens
# ---------------------------------------------------------------------------

class TestAllowedNextNumberTokens:
    def setup_method(self) -> None:
        _allowed_number_cache.clear()

    def test_empty_built_allows_digits_and_minus(self) -> None:
        allowed = allowed_next_number_tokens(VOCAB, "")
        # digits and minus sign should be allowed from empty start
        assert VOCAB.index("1") in allowed
        assert VOCAB.index("-") in allowed
        # leading dot must not be allowed
        assert VOCAB.index(".") not in allowed
        # non-numeric token must not be allowed
        assert VOCAB.index("abc") not in allowed

    def test_digit_built_allows_dot_and_digits(self) -> None:
        allowed = allowed_next_number_tokens(VOCAB, "1")
        assert VOCAB.index(".") in allowed
        assert VOCAB.index("2") in allowed

    def test_no_double_minus(self) -> None:
        allowed = allowed_next_number_tokens(VOCAB, "-1")
        assert VOCAB.index("-") not in allowed

    def test_no_double_dot(self) -> None:
        allowed = allowed_next_number_tokens(VOCAB, "1.")
        assert VOCAB.index(".") not in allowed

    def test_no_leading_zero_before_digit(self) -> None:
        # "0" + "1" → "01" → invalid leading zero
        vocab = ["", "0", "1", "."]
        allowed = allowed_next_number_tokens(vocab, "0")
        assert vocab.index("1") not in allowed
        assert vocab.index(".") in allowed

    def test_result_is_cached(self) -> None:
        _allowed_number_cache.clear()
        allowed_next_number_tokens(VOCAB, "5")
        assert "5" in _allowed_number_cache
        # Second call must hit the cache (same object)
        result1 = allowed_next_number_tokens(VOCAB, "5")
        result2 = allowed_next_number_tokens(VOCAB, "5")
        assert result1 is result2


# ---------------------------------------------------------------------------
# mask_logits_in_place
# ---------------------------------------------------------------------------

class TestMaskLogitsInPlace:
    def test_masks_non_allowed(self) -> None:
        logits = [1.0, 2.0, 3.0, 4.0]
        mask_logits_in_place(logits, {1, 3})
        assert logits[0] == NEG_INF
        assert logits[1] == 2.0
        assert logits[2] == NEG_INF
        assert logits[3] == 4.0

    def test_empty_allowed_masks_all(self) -> None:
        logits = [1.0, 2.0]
        mask_logits_in_place(logits, set())
        assert all(v == NEG_INF for v in logits)


# ---------------------------------------------------------------------------
# decode_number — mocked LLM
# ---------------------------------------------------------------------------

class MockLLM:
    """Minimal mock that returns a pre-set sequence of logit vectors."""

    def __init__(self, vocab: list[str], token_sequence: list[str]) -> None:
        self._vocab = vocab
        self._sequence = token_sequence
        self._call = 0

    def encode(self, text: str) -> Any:
        import torch
        return torch.tensor([[1, 2, 3]], dtype=torch.long)

    def decode(self, ids: Any) -> str:
        return ""

    def get_path_to_vocab_file(self) -> str:
        return ""

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        logits = [0.0] * len(self._vocab)
        if self._call < len(self._sequence):
            target_tok = self._sequence[self._call]
            if target_tok in self._vocab:
                logits[self._vocab.index(target_tok)] = 100.0
        self._call += 1
        return logits


class TestDecodeNumber:
    def setup_method(self) -> None:
        _allowed_number_cache.clear()

    def test_decodes_integer(self) -> None:
        # vocab idx: 0="", 1="4", 2="2", 3=".", 4="-", 5=" ", 6="\n", 7="abc"
        vocab2 = ["", "4", "2", ".", "-", " ", "\n", "abc"]
        llm2 = MockLLM(vocab2, ["4", "2"])
        result = decode_number(
            llm2, "add 4 and 2", id_to_token=vocab2, param_name="a", max_steps=2,
        )
        assert result == 42.0

    def test_returns_zero_on_empty(self) -> None:
        vocab = ["", "abc", " ", "\n"]  # no numeric tokens
        llm = MockLLM(vocab, [])
        result = decode_number(llm, "no number here", id_to_token=vocab)
        assert result == 0.0

    def test_trailing_dot_stripped(self) -> None:
        vocab = ["", "5", ".", " ", "\n", "abc"]
        # Forces "5" then "." — result should be 5.0, not ValueError
        llm = MockLLM(vocab, ["5", "."])
        result = decode_number(llm, "give me 5", id_to_token=vocab, max_steps=2)
        assert result == 5.0
