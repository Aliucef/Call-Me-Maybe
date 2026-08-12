from src.decoder.candidate_chooser import choose_from_candidates
from src.model.input_format import FunctionDef
from src.model.llm_protocol import LLMProtocol


def choose_function_name(
    llm: LLMProtocol,
    prompt_text: str,
    functions: list[FunctionDef],
    name_to_ids: dict[str, list[int]],
    verbose: bool = False,
) -> str:
    """Select the best matching function name using constrained greedy decoding."""
    lines = [
        f"- {f.name}: {f.description or ''} params={list(f.parameters.keys())}"
        for f in functions
    ]
    menu = "\n".join(lines)
    context = (
        "Choose the best function for the request.\n"
        "Available functions:\n"
        f"{menu}\n"
        f"Request: {prompt_text}\n"
        "Answer with the function name only:\n"
    )
    return choose_from_candidates(llm, context, name_to_ids, verbose=verbose, log_label="fn")
