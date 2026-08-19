"""Unit tests for function_chooser — LLM is mocked."""
from src.decoder.function_chooser import choose_function_name
from src.model.input_format import FunctionDef, ParameterSchema
from tests.conftest import MockLLM


def _fn(name: str) -> FunctionDef:
    return FunctionDef(
        name=name, description=f"do {name}",
        parameters={"x": ParameterSchema(type="number")},
    )


class TestChooseFunctionName:
    def test_returns_valid_name(self) -> None:
        functions = [_fn("fn_add"), _fn("fn_sub")]
        name_to_ids = {"fn_add": [10, 20, 30], "fn_sub": [40, 50, 60]}
        llm = MockLLM(vocab_size=100, preferred_token_per_step=[10, 20, 30])
        result = choose_function_name(llm, "add 1 and 2", functions, name_to_ids)
        assert result == "fn_add"

    def test_chooses_second_function(self) -> None:
        functions = [_fn("fn_add"), _fn("fn_sub")]
        name_to_ids = {"fn_add": [10, 20, 30], "fn_sub": [40, 50, 60]}
        llm = MockLLM(vocab_size=100, preferred_token_per_step=[40, 50, 60])
        result = choose_function_name(llm, "subtract 5 from 10", functions, name_to_ids)
        assert result == "fn_sub"

    def test_result_always_in_name_to_ids(self) -> None:
        functions = [_fn("fn_a"), _fn("fn_b"), _fn("fn_c")]
        name_to_ids = {"fn_a": [1, 2], "fn_b": [3, 4], "fn_c": [5, 6]}
        llm = MockLLM(vocab_size=20, preferred_token_per_step=[])
        result = choose_function_name(llm, "some prompt", functions, name_to_ids)
        assert result in name_to_ids

    def test_verbose_does_not_raise(self) -> None:
        functions = [_fn("fn_x")]
        name_to_ids = {"fn_x": [7, 8]}
        llm = MockLLM(vocab_size=20, preferred_token_per_step=[7, 8])
        result = choose_function_name(llm, "x", functions, name_to_ids, verbose=True)
        assert result == "fn_x"

    def test_single_function_is_returned(self) -> None:
        functions = [_fn("fn_only")]
        name_to_ids = {"fn_only": [5, 6, 7]}
        llm = MockLLM(vocab_size=20, preferred_token_per_step=[5, 6, 7])
        result = choose_function_name(llm, "do the thing", functions, name_to_ids)
        assert result == "fn_only"
