from src.model.llm_protocol import LLMProtocol

NEG_INF = -1e30
# A finite "negative infinity" stand-in. We can't use math.inf directly in a
# way that always plays nicely with float arithmetic downstream, so a very
# large negative number is used instead — small enough to never win an
# argmax against a real logit, but still a normal float.

# Cache allowed token sets per partial number string — the vocab is fixed per run.
# Key = the partial number text built so far (e.g. "", "2", "-", "1."),
# Value = the set of token ids allowed to follow it. Since the same partial
# strings recur across different prompts/parameters within one run (e.g.
# "" is the starting point every single time decode_number is called), this
# avoids re-scanning the whole vocabulary from scratch each time.
_allowed_number_cache: dict[str, set[int]] = {}


def mask_logits_in_place(logits: list[float], allowed: set[int]) -> None:
    """Set logits of all tokens not in allowed to NEG_INF in place."""
    # After this, taking max(range(len(logits)), key=logits.__getitem__)
    # is equivalent to restricting the argmax search to `allowed` only,
    # while still being able to reuse the same "find the max" code path
    # used for the unconstrained lookahead check.
    for i in range(len(logits)):
        if i not in allowed:
            logits[i] = NEG_INF


def allowed_next_number_tokens(id_to_token: list[str], built: str) -> set[int]:
    """Return the set of token IDs that keep the partial string a valid number prefix."""
    # Memoized on `built` — see _allowed_number_cache above.
    if built in _allowed_number_cache:
        return _allowed_number_cache[built]

    allowed: set[int] = set()
    # Scan the ENTIRE vocabulary (id_to_token is indexed by token id) and
    # test each token as a candidate continuation of `built`.
    for tid, tok in enumerate(id_to_token):
        if not tok:
            # Some vocab slots may be empty placeholders; skip them.
            continue
        cand = built + tok
        # BPE tokens occasionally carry a leading marker for whitespace;
        # lstrip() tolerates a genuine leading space in the decoded piece.
        s = cand.lstrip()
        if not s:
            continue
        # Reject anything that isn't purely digits, '-' or '.' — e.g. a
        # token like "and" or " the" is immediately disqualified, which is
        # what keeps the LLM from wandering off into free text.
        if not all(ch.isdigit() or ch in "-." for ch in s):
            continue
        # At most one minus sign, and only as the very first character
        # (rules out "2-3" or "--2").
        if s.count("-") > 1:
            continue
        if "-" in s and not s.startswith("-"):
            continue
        # At most one decimal point (rules out "1.2.3").
        if s.count(".") > 1:
            continue
        # Strip the optional leading '-' to check the numeric part alone.
        t = s[1:] if s.startswith("-") else s
        # Reject a decimal point with nothing before it (e.g. ".5" is not
        # allowed — the model must produce "0.5" instead).
        if t.startswith("."):
            continue
        # Reject leading zeros before another digit (e.g. "02"), but still
        # allow "0" alone and "0.5" (t[1] is '.' there, not a digit).
        if len(t) >= 2 and t[0] == "0" and t[1].isdigit():
            continue
        # Survived every rule: this token is a legal continuation.
        allowed.add(tid)

    _allowed_number_cache[built] = allowed
    return allowed


def decode_number(
    llm: LLMProtocol,
    prompt_text: str,
    *,
    id_to_token: list[str],
    param_name: str | None = None,
    already_extracted: dict[str, object] | None = None,
    max_steps: int = 12,
    verbose: bool = False,
) -> float:
    """Decode a numeric parameter using constrained greedy decoding.

    Only tokens that extend the partial result into a valid number literal
    are eligible at each step, guaranteeing a parseable float output.

    RUNNING EXAMPLE:
        prompt_text = "What is the sum of 2 and 3?"
        fn_add_numbers has parameters {"a": number, "b": number}.
        First call:  param_name="a", already_extracted={}          -> returns 2.0
        Second call: param_name="b", already_extracted={"a": 2.0}  -> returns 3.0
    """
    # Build the instruction line, e.g. 'Extract the number for parameter "a".'
    instr = (
        f'Extract the number for parameter "{param_name}".'
        if param_name else "Extract the number."
    )
    prior = ""
    if already_extracted:
        # For the "b" call this becomes: "Already extracted: a=2.0\n"
        # Giving the model previously decoded parameters as context helps it
        # avoid re-picking the same number for the next parameter.
        pairs = ", ".join(f'{k}={v}' for k, v in already_extracted.items())
        prior = f"Already extracted: {pairs}\n"
    # Full prompt, e.g. for param "a":
    #   'Extract the number for parameter "a".\n'
    #   'Request: What is the sum of 2 and 3?\n'
    #   'Answer with only the number:\n'
    context = (
        f"{instr}\n"
        f"{prior}"
        f"Request: {prompt_text}\n"
        "Answer with only the number:\n"
    )

    input_ids: list[int] = llm.encode(context).squeeze(0).tolist()
    built = ""  # the number literal assembled so far, e.g. "" -> "2"

    for step in range(max_steps):
        # Unconstrained forward pass — scores for every token in the vocab.
        logits = llm.get_logits_from_input_ids(input_ids)

        # Stop early when the model's top unconstrained choice is a stop token
        # (BPE newline Ċ, literal \n, or empty) — number is complete.
        # Example: after built="2", the model's single best next guess is
        # very likely a newline (it thinks the answer is done), so we stop
        # here instead of forcing it to keep emitting digits.
        best_overall = max(range(len(logits)), key=lambda i: logits[i])
        best_tok = id_to_token[best_overall]
        if not best_tok or "Ċ" in best_tok or "\n" in best_tok:
            break

        # Compute which tokens are legal continuations of `built` right now.
        allowed = allowed_next_number_tokens(id_to_token, built=built)
        if not allowed:
            # No legal continuation exists (shouldn't normally happen since
            # digits are always allowed) — bail out rather than looping.
            break

        # Mask out everything except the legal continuations, then take the
        # highest-scoring token among only those — this is the constrained
        # decoding step. Continuing the example: among {"0".."9", "-", "."}
        # (roughly), "2" scores highest because the prompt literally says "2".
        mask_logits_in_place(logits, allowed)
        next_id = max(range(len(logits)), key=lambda i: logits[i])
        piece = id_to_token[next_id]

        if verbose:
            print(f"  [num step {step}] built={repr(built + piece)}")

        built += piece            # e.g. "" + "2" -> "2"
        input_ids.append(next_id)  # commit the token so the next forward pass sees it

    # Post-processing: turn the assembled string into a float, defensively.
    s = built.strip()
    if s in {"", "-"}:
        # Nothing usable was decoded (e.g. immediate stop token) — default
        # to 0.0 rather than raising, per "no unhandled exceptions".
        return 0.0
    if s.endswith("."):
        # A trailing bare decimal point (e.g. "3.") isn't valid float()
        # input on some platforms' edge cases; drop it defensively.
        s = s[:-1]
    try:
        # Example: float("2") -> 2.0, float("3") -> 3.0
        return float(s)
    except ValueError:
        # Should be unreachable given the character-level constraints
        # above, but kept as a last-resort safety net.
        return 0.0
