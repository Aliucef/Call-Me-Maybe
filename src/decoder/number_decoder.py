import re
from collections import Counter

from src.decoder.candidate_chooser import choose_from_candidates
from src.model.llm_protocol import LLMProtocol

_NUMBER_LITERAL = re.compile(r"-?\d+(?:\.\d+)?")

_number_encode_cache: dict[str, list[int]] = {}


def extract_number_candidates(prompt_text: str) -> list[str]:
    """Return every number literal that appears verbatim in the prompt, in order."""
    return _NUMBER_LITERAL.findall(prompt_text)


def encode_number_candidate(llm: LLMProtocol, literal: str) -> list[int]:
    """Encode a numeric literal to token ids, reusing a cached result if seen before."""
    cached = _number_encode_cache.get(literal)
    if cached is not None:
        return cached
    ids: list[int] = llm.encode(literal).squeeze(0).tolist()
    _number_encode_cache[literal] = ids
    return ids


def choose_number(
    llm: LLMProtocol,
    prompt_text: str,
    *,
    param_name: str | None = None,
    already_extracted: dict[str, object] | None = None,
    verbose: bool = False,
) -> float | None:
    """Pick a number that literally appears in the prompt, or None if none is available."""
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
        for s, ids in ((s, encode_number_candidate(llm, s)) for s in remaining)
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
    """Resolve a numeric parameter, defaulting to 0.0 if none is found."""
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
