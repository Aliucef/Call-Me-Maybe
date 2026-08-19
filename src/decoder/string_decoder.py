import re

from src.decoder.candidate_chooser import choose_from_candidates
from src.model.llm_protocol import LLMProtocol

_QUOTED_SPAN = re.compile(r"'([^']*)'|\"([^\"]*)\"")
_CONCEPT_WORD = re.compile(r"[A-Za-z]+")

_REGEX_CONCEPTS: dict[str, str] = {
    "number": r"\d+", "numbers": r"\d+", "digit": r"\d+", "digits": r"\d+",
    "vowel": r"[aeiouAEIOU]", "vowels": r"[aeiouAEIOU]",
    "consonant": r"[^aeiouAEIOU\s]", "consonants": r"[^aeiouAEIOU\s]",
    "letter": r"[A-Za-z]", "letters": r"[A-Za-z]+",
    "whitespace": r"\s+", "space": r"\s+", "spaces": r"\s+",
    "punctuation": r"[^\w\s]",
    "uppercase": r"[A-Z]+", "lowercase": r"[a-z]+",
}


def extract_quoted_spans(prompt_text: str) -> list[str]:
    """Return every quoted substring in the prompt, in order of appearance."""
    return [
        m.group(1) if m.group(1) is not None else m.group(2)
        for m in _QUOTED_SPAN.finditer(prompt_text)
    ]


def extract_regex_pattern_candidates(prompt_text: str) -> list[str]:
    """Return built-in regex patterns for recognized concept words in the prompt."""
    found: list[str] = []
    for word in _CONCEPT_WORD.findall(prompt_text):
        pattern = _REGEX_CONCEPTS.get(word.lower())
        if pattern and pattern not in found:
            found.append(pattern)
    return found


def extract_trailing_word_candidate(prompt_text: str) -> list[str]:
    """Return the last standalone unquoted word in the prompt, if any."""
    unquoted = _QUOTED_SPAN.sub(" ", prompt_text)
    words = _CONCEPT_WORD.findall(unquoted)
    return [words[-1]] if len(words) >= 2 else []


def choose_string(
    llm: LLMProtocol,
    prompt_text: str,
    *,
    param_name: str | None = None,
    already_extracted: dict[str, object] | None = None,
    verbose: bool = False,
) -> str | None:
    """Pick a value from quoted spans, known regex concepts, or the trailing word."""
    used_values = {v for v in (already_extracted or {}).values() if isinstance(v, str)}
    unique_spans = list(dict.fromkeys(extract_quoted_spans(prompt_text)))
    remaining = [s for s in unique_spans if s not in used_values]
    if not remaining:
        concept_patterns = extract_regex_pattern_candidates(prompt_text)
        remaining = [p for p in concept_patterns if p not in used_values]
    if not remaining:
        trailing_word = extract_trailing_word_candidate(prompt_text)
        remaining = [w for w in trailing_word if w not in used_values]
    if not remaining:
        return None
    remaining.sort(key=len, reverse=True)

    candidates_by_key = {
        s: ids
        for s, ids in ((s, llm.encode(s).squeeze(0).tolist()) for s in remaining)
        if ids
    }
    if not candidates_by_key:
        return None

    label = f'"{param_name}"' if param_name else "the string value"
    listing = ", ".join(repr(s) for s in remaining)
    prior = ""
    if already_extracted:
        pairs = ", ".join(f"{k}={v!r}" for k, v in already_extracted.items())
        prior = f"Already extracted: {pairs}\n"
    context = (
        f"Pick the value for parameter {label} from the quoted values below.\n"
        f"Quoted values in the request: {listing}\n"
        f"{prior}"
        f"Request: {prompt_text}\n"
        "Answer with only one of the values listed above, no quotes:\n"
    )

    return choose_from_candidates(
        llm, context, candidates_by_key, verbose=verbose, log_label="str-choice",
    )


def resolve_string(
    llm: LLMProtocol,
    prompt_text: str,
    *,
    param_name: str | None = None,
    already_extracted: dict[str, object] | None = None,
    verbose: bool = False,
) -> str:
    """Resolve a string parameter, defaulting to "" if none is found."""
    value = choose_string(
        llm,
        prompt_text,
        param_name=param_name,
        already_extracted=already_extracted,
        verbose=verbose,
    )
    if value is not None:
        return value
    label = f'"{param_name}"' if param_name else "a required string"
    print(f'[ERROR] Could not find {label} in the request: {prompt_text!r} -- defaulting to ""')
    return ""
