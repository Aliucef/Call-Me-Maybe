import re
from collections import Counter

from src.decoder.candidate_chooser import choose_from_candidates
from src.model.llm_protocol import LLMProtocol

_NUMBER_LITERAL = re.compile(r"-?\d+(?:\.\d+)?")


def extract_number_candidates(prompt_text: str) -> list[str]:
    """Return every number literal that appears verbatim in the prompt, in order."""
    return _NUMBER_LITERAL.findall(prompt_text)


def choose_number(
    llm: LLMProtocol,
    prompt_text: str,
    *,
    param_name: str | None = None,
    already_extracted: dict[str, object] | None = None,
    verbose: bool = False,
) -> float | None:
    """Pick a number that literally appears in the prompt via constrained
    candidate-narrowing decoding (same mechanism as choose_function_name).

    Restricts the choice to the actual numbers found in the prompt — the
    model can never hallucinate a value that isn't there, and picking
    between a handful of known candidates is an easier task than free
    generation. Returns None if the prompt has no number-like substrings,
    or if every occurrence of every number mentioned has already been
    claimed by an earlier parameter (e.g. "sum of 2 and ?" only has one
    number for two parameters — the second must not silently repeat the
    first).
    """
    candidate_strs = extract_number_candidates(prompt_text)
    if not candidate_strs:
        return None

    # Count *occurrences*, not just distinct values, so "sum of 2 and 2"
    # (two real occurrences of 2) still correctly allows both parameters
    # to get 2.0, while "sum of 2 and ?" (one occurrence) correctly runs
    # out of candidates for the second parameter instead of reusing the
    # first value just because it's the only one that exists.
    prompt_counts = Counter(float(s) for s in candidate_strs)
    used_counts = Counter(
        float(v)
        for v in (already_extracted or {}).values()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    )

    unique_strs = list(dict.fromkeys(candidate_strs))
    remaining = [s for s in unique_strs if prompt_counts[float(s)] > used_counts[float(s)]]
    if not remaining:
        return None

    candidates_by_key = {
        s: ids
        for s, ids in ((s, llm.encode(s).squeeze(0).tolist()) for s in remaining)
        if ids
    }
    if not candidates_by_key:
        return None

    label = f'"{param_name}"' if param_name else "the number"
    listing = ", ".join(remaining)

    prior = ""
    if already_extracted:
        pairs = ", ".join(f"{k}={v}" for k, v in already_extracted.items())
        prior = f"Already extracted: {pairs}\n"
    context = (
        f"Pick the number for parameter {label} from the numbers mentioned below.\n"
        f"Numbers mentioned in the request: {listing}\n"
        f"{prior}"
        f"Request: {prompt_text}\n"
        "Answer with only one of the numbers listed above:\n"
    )

    chosen = choose_from_candidates(
        llm, context, candidates_by_key, verbose=verbose, log_label="num-choice",
    )
    try:
        return float(chosen)
    except ValueError:
        return None


def resolve_number(
    llm: LLMProtocol,
    prompt_text: str,
    *,
    param_name: str | None = None,
    already_extracted: dict[str, object] | None = None,
    verbose: bool = False,
) -> float:
    """Resolve a numeric parameter end to end.

    Tries choose_number(), which only ever returns a value that literally
    appears in the prompt. If there is no legitimate number for this
    parameter -- no number-like substring in the prompt at all, or every
    occurrence already claimed by an earlier parameter (e.g. "sum of 2
    and ?" has only one number for two parameters) -- there is nothing to
    extract, this prints a clear error and defaults to 0.0, since the
    output schema requires every parameter present with a valid type even
    when the request is genuinely underspecified.
    """
    value = choose_number(
        llm,
        prompt_text,
        param_name=param_name,
        already_extracted=already_extracted,
        verbose=verbose,
    )
    if value is not None:
        return value
    label = f'"{param_name}"' if param_name else "a required number"
    print(f"[ERROR] Could not find {label} in the request: {prompt_text!r} -- defaulting to 0.0")
    return 0.0
