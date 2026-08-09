import re

from src.decoder.candidate_chooser import choose_from_candidates
from src.model.llm_protocol import LLMProtocol

_QUOTED_SPAN = re.compile(r"'([^']*)'|\"([^\"]*)\"")
_CONCEPT_WORD = re.compile(r"[A-Za-z]+")

# Built-in regex patterns for common textual-category concepts. This is
# domain knowledge about regex syntax itself -- the same kind of built-in
# grammar allowed_next_number_tokens() already relies on for number
# literals -- not a lookup tied to a specific function name or exact
# prompt wording, so it doesn't reintroduce the "hardcoded prompt"
# heuristic the subject forbids for function selection. A regex pattern
# like "\d+" never appears verbatim in a prompt like "replace all
# numbers", so no amount of extraction from the prompt text alone can
# produce it -- it has to come from knowing what "numbers" means in regex.
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
    """Return built-in regex patterns for any recognized textual-category
    concept word (e.g. "numbers", "vowels") mentioned in the prompt, in
    order of appearance."""
    found: list[str] = []
    for word in _CONCEPT_WORD.findall(prompt_text):
        pattern = _REGEX_CONCEPTS.get(word.lower())
        if pattern and pattern not in found:
            found.append(pattern)
    return found


def extract_trailing_word_candidate(prompt_text: str) -> list[str]:
    """Return the last standalone (non-quoted) word in the prompt, if any.

    A last-resort single candidate for a literal value that appears
    verbatim but unquoted at the end of the sentence (e.g. a replacement
    token in "... with asterisks"). Natural-language requests commonly put
    the sentence's final complement last, so this is a general positional
    signal rather than one tied to a specific keyword or function.

    Requires at least two leftover words. "Last" is only a meaningful
    signal relative to something earlier in the sentence -- with a single
    leftover word (e.g. a bare "Greet" with no name at all), there is
    nothing to distinguish an instruction word from an actual payload, so
    this deliberately returns nothing rather than guessing.
    """
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
    """Pick a value that appears quoted in the prompt via constrained
    candidate-narrowing decoding (same mechanism as choose_function_name /
    choose_number).

    Tries three tiers, each only consulted if the previous one is empty:
    1. Quoted substrings — values guaranteed to appear verbatim.
    2. A small built-in library of regex patterns for common textual
       category concepts (see _REGEX_CONCEPTS), when the prompt mentions
       one — a regex pattern is never literal prompt text, so it can only
       come from domain knowledge, not extraction.
    3. The prompt's trailing unquoted word — a positional fallback for a
       literal value that's unquoted (e.g. a name, or a replacement token
       at the end of the sentence).
    Returns None if none of the three yield anything.
    """
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

    # Longer spans tend to be the text being operated on (e.g. source_string
    # in a substitution call); shorter ones tend to be find/replace
    # targets. Sorting longest-first lets the model's own "pick the first
    # listed option" bias work for us instead of against us, the same
    # trick used in choose_number for already-used values.
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
    """Resolve a string parameter end to end.

    Tries choose_string(), which only ever returns a value that literally
    appears in the prompt (quoted, a known regex-pattern concept, or the
    trailing word). If none of those tiers found anything grounded to
    extract, this prints a clear error and defaults to an empty string,
    since the output schema requires every parameter present even when the
    request is genuinely underspecified.
    """
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
