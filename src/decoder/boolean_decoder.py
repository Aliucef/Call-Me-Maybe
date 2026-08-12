from src.decoder.candidate_chooser import choose_from_candidates
from src.model.llm_protocol import LLMProtocol

_TRUE = "true"
_FALSE = "false"

# Generic on/off, enable/disable few-shot examples -- not tied to any
# specific function name or prompt wording, just priming the model on the
# true/false judgment task itself. A 0.6B model answers this kind of
# binary intent question far more reliably with a couple of worked
# examples than with the bare instruction alone.
_FEW_SHOT_EXAMPLES = (
    "Examples:\n"
    "Request: Turn on the lights\nAnswer: true\n"
    "Request: Turn off the lights\nAnswer: false\n"
    "Request: Enable notifications\nAnswer: true\n"
    "Request: Disable notifications\nAnswer: false\n"
    "Request: Please turn off the fan\nAnswer: false\n"
    "Request: Please turn on the fan\nAnswer: true\n"
)


def resolve_boolean(
    llm: LLMProtocol,
    prompt_text: str,
    *,
    param_name: str | None = None,
    already_extracted: dict[str, object] | None = None,
    verbose: bool = False,
) -> bool:
    """Resolve a boolean parameter by choosing between "true" and "false"."""
    label = f'"{param_name}"' if param_name else "the boolean value"
    prior = ""
    if already_extracted:
        pairs = ", ".join(f"{k}={v!r}" for k, v in already_extracted.items())
        prior = f"Already extracted: {pairs}\n"
    context = (
        f"Decide the true or false value for parameter {label} based on the "
        "request below.\n"
        f"{_FEW_SHOT_EXAMPLES}"
        f"{prior}"
        f"Request: {prompt_text}\n"
        "Answer:\n"
    )

    candidates_by_key = {
        word: ids
        for word, ids in (
            (word, llm.encode(word).squeeze(0).tolist()) for word in (_TRUE, _FALSE)
        )
        if ids
    }
    if not candidates_by_key:
        error_label = f'"{param_name}"' if param_name else "a required boolean"
        print(
            f"[ERROR] Could not encode true/false candidates for {error_label}"
            " -- defaulting to False"
        )
        return False

    chosen = choose_from_candidates(
        llm, context, candidates_by_key, verbose=verbose, log_label="bool-choice",
    )
    return chosen == _TRUE
