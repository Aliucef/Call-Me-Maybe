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

# A tiny stand-in "vocabulary": a list of token strings indexed by id, just
# like the real id_to_token built in src/main.py from vocab.json — but with
# only 10 entries instead of ~150k, so tests are fast and results are
# trivial to reason about by hand.
VOCAB = ["", "1", "2", "3", ".", "-", "0", "abc", "10", " "]


def _ids(tokens: list[str]) -> set[int]:
    # Convenience: turn a list of token strings into the set of ids VOCAB
    # assigns them, e.g. _ids(["1", "-"]) -> {1, 5}.
    return {VOCAB.index(t) for t in tokens}


# ---------------------------------------------------------------------------
# allowed_next_number_tokens
# ---------------------------------------------------------------------------

class TestAllowedNextNumberTokens:
    def setup_method(self) -> None:
        # allowed_next_number_tokens caches by the `built` string globally
        # (_allowed_number_cache), so tests must clear it first — otherwise
        # a result cached under e.g. "1" by one test would leak into
        # another test using a different fake VOCAB for the same key.
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
        # After "1" is already built, both another digit ("2" -> "12") and
        # a decimal point ("." -> "1.") are valid continuations.
        allowed = allowed_next_number_tokens(VOCAB, "1")
        assert VOCAB.index(".") in allowed
        assert VOCAB.index("2") in allowed

    def test_no_double_minus(self) -> None:
        # "-1" + "-" would be "-1-", which is not a valid number — the
        # allowed set for built="-1" must therefore exclude "-".
        allowed = allowed_next_number_tokens(VOCAB, "-1")
        assert VOCAB.index("-") not in allowed

    def test_no_double_dot(self) -> None:
        # "1." + "." would be "1..", also invalid.
        allowed = allowed_next_number_tokens(VOCAB, "1.")
        assert VOCAB.index(".") not in allowed

    def test_no_leading_zero_before_digit(self) -> None:
        # "0" + "1" → "01" → invalid leading zero
        vocab = ["", "0", "1", "."]
        allowed = allowed_next_number_tokens(vocab, "0")
        assert vocab.index("1") not in allowed
        # ...but "0" + "." -> "0." is fine, since a decimal point after a
        # single leading zero is a normal number ("0.5").
        assert vocab.index(".") in allowed

    def test_result_is_cached(self) -> None:
        _allowed_number_cache.clear()
        allowed_next_number_tokens(VOCAB, "5")
        assert "5" in _allowed_number_cache
        # Second call must hit the cache (same object)
        # `is` checks object identity, not just equal contents — proving
        # the second and third calls returned the exact cached set object
        # rather than recomputing and rebuilding an equal-but-different one.
        result1 = allowed_next_number_tokens(VOCAB, "5")
        result2 = allowed_next_number_tokens(VOCAB, "5")
        assert result1 is result2


# ---------------------------------------------------------------------------
# mask_logits_in_place
# ---------------------------------------------------------------------------

class TestMaskLogitsInPlace:
    def test_masks_non_allowed(self) -> None:
        logits = [1.0, 2.0, 3.0, 4.0]
        # Only indices 1 and 3 are allowed; 0 and 2 should be knocked down
        # to NEG_INF so they can never win a subsequent argmax.
        mask_logits_in_place(logits, {1, 3})
        assert logits[0] == NEG_INF
        assert logits[1] == 2.0
        assert logits[2] == NEG_INF
        assert logits[3] == 4.0

    def test_empty_allowed_masks_all(self) -> None:
        # Edge case: nothing is allowed -> every entry gets masked, which
        # is what makes decode_number's "if not allowed: break" path safe
        # even if this function were called with an empty allowed set.
        logits = [1.0, 2.0]
        mask_logits_in_place(logits, set())
        assert all(v == NEG_INF for v in logits)


# ---------------------------------------------------------------------------
# decode_number — mocked LLM
# ---------------------------------------------------------------------------

class MockLLM:
    """Minimal mock that returns a pre-set sequence of logit vectors.

    Same idea as MockLLM in test_function_chooser.py: instead of a real
    model, this is handed an exact scripted sequence of token *strings* it
    should prefer at each successive call to get_logits_from_input_ids, so
    decode_number()'s constrained loop can be driven deterministically
    step by step without needing torch/transformers to actually run.
    """

    def __init__(self, vocab: list[str], token_sequence: list[str]) -> None:
        self._vocab = vocab
        self._sequence = token_sequence
        self._call = 0

    def encode(self, text: str) -> Any:
        # Only imported here (not at module level) since this mock is the
        # one place in the test file that needs a real tensor-shaped
        # return value; the prompt text itself is ignored.
        import torch
        return torch.tensor([[1, 2, 3]], dtype=torch.long)

    def decode(self, ids: Any) -> str:
        return ""  # unused by decode_number(), present only to satisfy LLMProtocol

    def get_path_to_vocab_file(self) -> str:
        return ""  # unused by decode_number(), present only to satisfy LLMProtocol

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        logits = [0.0] * len(self._vocab)
        if self._call < len(self._sequence):
            target_tok = self._sequence[self._call]
            if target_tok in self._vocab:
                # Boost this step's scripted token so it wins both the
                # "unconstrained top choice" check (used to detect a stop
                # signal) and the constrained masked argmax.
                logits[self._vocab.index(target_tok)] = 100.0
        self._call += 1
        return logits


class TestDecodeNumber:
    def setup_method(self) -> None:
        _allowed_number_cache.clear()

    def test_decodes_integer(self) -> None:
        # vocab idx: 0="", 1="4", 2="2", 3=".", 4="-", 5=" ", 6="\n", 7="abc"
        vocab2 = ["", "4", "2", ".", "-", " ", "\n", "abc"]
        # Script the model to emit "4" then "2" — decode_number should
        # build the string "42" and parse it as a float.
        llm2 = MockLLM(vocab2, ["4", "2"])
        result = decode_number(
            llm2, "add 4 and 2", id_to_token=vocab2, param_name="a", max_steps=2,
        )
        assert result == 42.0

    def test_returns_zero_on_empty(self) -> None:
        vocab = ["", "abc", " ", "\n"]  # no numeric tokens
        # With an empty preferred sequence, the mock always returns
        # all-zero logits, so the "best_overall" token is whatever index 0
        # happens to be ("" here) — decode_number must default to 0.0
        # rather than crashing when nothing numeric gets built.
        llm = MockLLM(vocab, [])
        result = decode_number(llm, "no number here", id_to_token=vocab)
        assert result == 0.0

    def test_trailing_dot_stripped(self) -> None:
        vocab = ["", "5", ".", " ", "\n", "abc"]
        # Forces "5" then "." — result should be 5.0, not ValueError
        # This exercises the `if s.endswith("."): s = s[:-1]` cleanup step,
        # since float("5.") would actually work in plain Python, but this
        # confirms decode_number normalizes it the same way regardless.
        llm = MockLLM(vocab, ["5", "."])
        result = decode_number(llm, "give me 5", id_to_token=vocab, max_steps=2)
        assert result == 5.0
