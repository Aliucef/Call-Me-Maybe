from src.model.llm_protocol import LLMProtocol


def decode_string(
    llm: LLMProtocol,
    prompt_text: str,
    *,
    id_to_token: list[str],
    param_name: str | None = None,
    already_extracted: dict[str, object] | None = None,
    max_steps: int = 30,
    verbose: bool = False,
) -> str:
    """Decode a string parameter using constrained greedy decoding.

    Tokens containing a newline character are excluded at every step so the
    model is forced to stay on a single output line.  Decoding stops as soon
    as a newline token would be the best choice (i.e. the model signals it is
    done) or max_steps is reached.

    RUNNING EXAMPLE:
        prompt_text = "Greet shrek"
        fn_greet has parameters {"name": string}.
        Call: param_name="name", already_extracted={} -> returns "shrek"
    """
    # Human-readable label for the parameter being extracted, e.g. '"name"'.
    label = f'"{param_name}"' if param_name else "the string value"
    prior = ""
    if already_extracted:
        # Same idea as in number_decoder.py: show previously decoded
        # parameters (with repr() so strings are quoted) as extra context.
        pairs = ", ".join(f'{k}={v!r}' for k, v in already_extracted.items())
        prior = f"Already extracted: {pairs}\n"
    # Full prompt built for this example:
    #   'Extract the value for "name".\n'
    #   'Request: Greet shrek\n'
    #   'Answer with only the value, no quotes:\n'
    context = (
        f"Extract the value for {label}.\n"
        f"{prior}"
        f"Request: {prompt_text}\n"
        "Answer with only the value, no quotes:\n"
    )

    input_ids: list[int] = llm.encode(context).squeeze(0).tolist()
    # Unlike decode_number (which builds a string directly), here we keep
    # the raw token ids and let llm.decode() reconstruct the text at the
    # end — safer for arbitrary text since BPE token pieces don't always
    # concatenate cleanly as plain Python string addition would suggest.
    built_ids: list[int] = []

    for step in range(max_steps):
        logits = llm.get_logits_from_input_ids(input_ids)

        # Tokens without a newline are allowed; find the best overall token to
        # detect when the model wants to stop (newline = done signal).
        # Example: after committing the token(s) for "shrek", the model's
        # own top unconstrained guess for what comes next is a newline
        # (it considers the answer complete), so we stop there.
        best_overall = max(range(len(logits)), key=lambda i: logits[i])
        best_tok = id_to_token[best_overall]
        # Ċ is the BPE encoding of \n used by Qwen / GPT-2 tokenizers
        if not best_tok or "Ċ" in best_tok or "\n" in best_tok:
            break

        # Build the allowed set fresh every step (unlike the number decoder,
        # this doesn't depend on `built`, only on which tokens are
        # newline-free, so it's the same set every iteration — no caching
        # needed, though it is recomputed each loop for simplicity).
        allowed = {
            tid
            for tid, tok in enumerate(id_to_token)
            if tok and "Ċ" not in tok and "\n" not in tok
        }
        if not allowed:
            break

        # Pick the highest-scoring token among the newline-free ones.
        # Example: at step 0, among all non-newline tokens, the one
        # spelling "shrek" (or its first BPE piece) scores highest because
        # it's copied verbatim from the prompt "Greet shrek".
        next_id = max(allowed, key=lambda i: logits[i])
        piece = id_to_token[next_id]

        if verbose:
            print(f"  [str step {step}] token={repr(piece)}")

        built_ids.append(next_id)
        input_ids.append(next_id)

    # Use the public decode() to convert BPE token IDs back to a proper string
    # Example: built_ids might decode to "shrek" or " shrek" depending on
    # tokenization; .strip() below removes any incidental leading space.
    result = llm.decode(built_ids).strip()
    # Strip any surrounding quotes the model may produce (handles partial too)
    # e.g. if the model ignored "no quotes" and produced '"shrek"' or
    # even just a stray leading/trailing quote character.
    if result.startswith(('"', "'")):
        result = result[1:]
    if result.endswith(('"', "'")):
        result = result[:-1]
    return result.strip()
