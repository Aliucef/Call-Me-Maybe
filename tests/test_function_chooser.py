"""Unit tests for function_chooser — LLM is mocked."""
import torch
from src.decoder.function_chooser import choose_function_name
from src.model.input_format import FunctionDef, ParameterSchema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fn(name: str) -> FunctionDef:
    # Minimal FunctionDef builder for tests — description and a single
    # dummy "x": number parameter are irrelevant to choose_function_name(),
    # which only reads .name/.description/.parameters for the prompt text
    # it builds, so any placeholder values are fine here.
    return FunctionDef(
        name=name, description=f"do {name}",
        parameters={"x": ParameterSchema(type="number")},
    )


class MockLLM:
    """Returns logits that favour a pre-chosen token at each decoding step.

    This stands in for llm_sdk.Small_LLM_Model in tests: instead of
    actually running a neural network, it's handed a scripted sequence of
    "preferred" token ids up front (preferred_token_per_step) and simply
    hands back a logits vector where that one token scores 100.0 and
    everything else scores 0.0 — deterministically steering
    choose_function_name()'s constrained argmax without needing a real
    model or GPU, and making the test's expected outcome exact and
    reproducible.
    """

    def __init__(self, vocab_size: int, preferred_token_per_step: list[int]) -> None:
        self._vocab_size = vocab_size
        self._preferred = preferred_token_per_step
        self._call = 0  # counts calls to get_logits_from_input_ids, i.e. the decoding step

    def encode(self, text: str) -> torch.Tensor:
        # The actual content of the prompt text doesn't matter for these
        # tests (the mock ignores it entirely) — just needs to look like a
        # real encode() result: a [1, seq_len] tensor of ids.
        return torch.tensor([[0, 1, 2]], dtype=torch.long)

    def decode(self, ids: object) -> str:
        return ""  # unused by choose_function_name(), present only to satisfy LLMProtocol

    def get_path_to_vocab_file(self) -> str:
        return ""  # unused by choose_function_name(), present only to satisfy LLMProtocol

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        # Start with a flat, all-zero score for every token in the vocab...
        logits = [0.0] * self._vocab_size
        if self._call < len(self._preferred):
            # ...then boost this step's scripted "correct" token so high
            # (100.0) that it always wins the constrained argmax, as long
            # as it's actually in the `allowed` set that step.
            logits[self._preferred[self._call]] = 100.0
        self._call += 1
        return logits


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestChooseFunctionName:
    def test_returns_valid_name(self) -> None:
        functions = [_fn("fn_add"), _fn("fn_sub")]
        # Token IDs for fn_add: [10, 20, 30], fn_sub: [40, 50, 60]
        name_to_ids = {"fn_add": [10, 20, 30], "fn_sub": [40, 50, 60]}
        # Guide logits: step 0 → 10, step 1 → 20, step 2 → 30
        # These are exactly fn_add's token ids in order, so the mock forces
        # the constrained decoder down the fn_add path at every step.
        llm = MockLLM(vocab_size=100, preferred_token_per_step=[10, 20, 30])
        result = choose_function_name(llm, "add 1 and 2", functions, name_to_ids)
        assert result == "fn_add"

    def test_chooses_second_function(self) -> None:
        functions = [_fn("fn_add"), _fn("fn_sub")]
        name_to_ids = {"fn_add": [10, 20, 30], "fn_sub": [40, 50, 60]}
        # Mirror of the previous test, but scripted to prefer fn_sub's ids
        # instead — proves the function isn't hardcoded to always pick the
        # first candidate.
        llm = MockLLM(vocab_size=100, preferred_token_per_step=[40, 50, 60])
        result = choose_function_name(llm, "subtract 5 from 10", functions, name_to_ids)
        assert result == "fn_sub"

    def test_result_always_in_name_to_ids(self) -> None:
        functions = [_fn("fn_a"), _fn("fn_b"), _fn("fn_c")]
        name_to_ids = {"fn_a": [1, 2], "fn_b": [3, 4], "fn_c": [5, 6]}
        # Logits all zero → picks whichever has highest id at each step
        # (max() breaks ties by the last-seen maximum among equal scores),
        # exercising the "no scripted preference" / tie-break path rather
        # than a specific target — the only real assertion is that
        # whatever comes out is still one of the legitimate candidates,
        # i.e. the function never invents a name outside name_to_ids.
        llm = MockLLM(vocab_size=20, preferred_token_per_step=[])
        result = choose_function_name(llm, "some prompt", functions, name_to_ids)
        assert result in name_to_ids

    def test_verbose_does_not_raise(self) -> None:
        functions = [_fn("fn_x")]
        name_to_ids = {"fn_x": [7, 8]}
        llm = MockLLM(vocab_size=20, preferred_token_per_step=[7, 8])
        # verbose=True exercises the debug-print branch inside the
        # decoding loop; this test only checks it doesn't crash and still
        # returns the right answer, not what gets printed.
        result = choose_function_name(llm, "x", functions, name_to_ids, verbose=True)
        assert result == "fn_x"

    def test_single_function_is_returned(self) -> None:
        functions = [_fn("fn_only")]
        name_to_ids = {"fn_only": [5, 6, 7]}
        # With only one candidate, `allowed` and `candidates` never really
        # narrow anything — this covers the trivial single-name case still
        # working correctly end to end.
        llm = MockLLM(vocab_size=20, preferred_token_per_step=[5, 6, 7])
        result = choose_function_name(llm, "do the thing", functions, name_to_ids)
        assert result == "fn_only"
