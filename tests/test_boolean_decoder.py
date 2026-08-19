from src.decoder.boolean_decoder import resolve_boolean
from tests.conftest import MockLLM


class TestResolveBoolean:
    def test_resolves_true(self) -> None:
        llm = MockLLM()
        llm.prefer("true")
        assert resolve_boolean(llm, "Turn on the fan", param_name="state") is True

    def test_resolves_false(self) -> None:
        llm = MockLLM()
        llm.prefer("false")
        assert resolve_boolean(llm, "Turn off the fan", param_name="state") is False

    def test_includes_already_extracted_context_without_raising(self) -> None:
        llm = MockLLM()
        llm.prefer("true")
        result = resolve_boolean(
            llm, "Enable notifications", param_name="state",
            already_extracted={"user": "john"},
        )
        assert result is True

    def test_verbose_does_not_raise(self) -> None:
        llm = MockLLM()
        llm.prefer("false")
        result = resolve_boolean(llm, "Disable notifications", verbose=True)
        assert result is False
