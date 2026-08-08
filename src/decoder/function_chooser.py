from src.model.input_format import FunctionDef
from src.model.llm_protocol import LLMProtocol


def choose_function_name(
    llm: LLMProtocol,
    prompt_text: str,
    functions: list[FunctionDef],
    name_to_ids: dict[str, list[int]],
    verbose: bool = False,
) -> str:
    """Select the best matching function name using constrained greedy decoding.

    At each step only tokens that continue a valid candidate function name
    are eligible, so the result is always a name from name_to_ids.

    RUNNING EXAMPLE used in the comments below:
        prompt_text = "What is the sum of 2 and 3?"
        functions   = [fn_add_numbers, fn_greet, fn_reverse_string,
                       fn_get_square_root, fn_substitute_string_with_regex]
        Expected result: "fn_add_numbers"
    """
    # Build one line per function describing it to the model, e.g.:
    # "- fn_add_numbers: Add two numbers together and return their sum. params=['a', 'b']"
    # "- fn_greet: Generate a greeting message for a person by name. params=['name']"
    # This gives the LLM enough context to judge which function fits the prompt.
    lines = [
        f"- {f.name}: {f.description or ''} params={list(f.parameters.keys())}"
        for f in functions
    ]
    menu = "\n".join(lines)
    # Full prompt fed to the model. Note it ends right after "Answer with the
    # function name only:\n" — the model's next tokens are what we decode.
    context = (
        "Choose the best function for the request.\n"
        "Available functions:\n"
        f"{menu}\n"
        f"Request: {prompt_text}\n"
        "Answer with the function name only:\n"
    )

    # Tokenize the whole context into a flat list[int] of token ids — this
    # is the running sequence we will keep appending decoded tokens to.
    input_ids: list[int] = llm.encode(context).squeeze(0).tolist()
    # `candidates` starts as every possible function name -> its token-id
    # sequence, e.g. {"fn_add_numbers": [16, 1997, ...], "fn_greet": [16, 6820, ...], ...}
    # (exact ids depend on the tokenizer; illustrative here.)
    # It shrinks every iteration as tokens get committed and rule out names
    # that no longer match what's been generated so far.
    candidates = dict(name_to_ids)
    # `pos` is the index into each candidate's token list we're currently
    # deciding — i.e. how many tokens of the function name have been
    # committed so far.
    pos = 0

    while True:
        # For every candidate still alive, look at the token id it expects
        # at position `pos`. `allowed` is the union of those ids — i.e. the
        # only tokens the model is permitted to output next.
        # Step 0 example: every name starts with the "fn_" prefix tokens, so
        # allowed at pos=0 is likely just {token_id_of("fn")} or similar —
        # all five candidates agree here, so nothing is eliminated yet.
        allowed = {ids[pos] for ids in candidates.values() if pos < len(ids)}
        if not allowed:
            # No candidate has a token left at this position — every
            # remaining candidate name has already been fully generated.
            # Falls through to the tie-break logic after the loop.
            break

        # One forward pass through the model: logits is a list[float] the
        # size of the whole vocabulary, one raw score per possible token —
        # this is the *unconstrained* distribution over every token, e.g.
        # the model might rank "add" highly given "sum of 2 and 3" appears
        # in the prompt, even before we restrict anything.
        logits = llm.get_logits_from_input_ids(input_ids)
        # This is the actual constraint: instead of taking the model's
        # global argmax, we only consider tokens in `allowed` and take the
        # highest-scoring one among those. Continuing the example, once the
        # prompt mentions "sum", "add" (part of fn_add_numbers) should score
        # far higher than "greet", "reverse", "get_square_root", etc.,
        # among the allowed first-token candidates.
        next_id = max(allowed, key=lambda tid: logits[tid])

        if verbose:
            # Debug view: show the top-3 candidate names ranked by the
            # score of their token at this position, and which token id
            # actually got committed.
            top = sorted(
                candidates,
                key=lambda n: logits[candidates[n][pos]] if pos < len(candidates[n]) else -1e30,
                reverse=True,
            )[:3]
            print(f"  [fn step {pos}] candidates={top} -> token_id={next_id}")

        # Commit the chosen token to the running sequence — the next
        # forward pass will see it as part of the context.
        input_ids.append(next_id)
        # Keep only the candidates whose token at `pos` matches what we
        # just committed. E.g. after committing the token for "add", every
        # candidate other than "fn_add_numbers" is eliminated in one shot
        # (fn_greet, fn_reverse_string, fn_get_square_root,
        # fn_substitute_string_with_regex all disagree at this position).
        candidates = {
            name: ids
            for name, ids in candidates.items()
            if pos < len(ids) and ids[pos] == next_id
        }
        pos += 1

        if len(candidates) == 1:
            # Exactly one candidate name survives. If we've also consumed
            # every one of its tokens (pos reached the end of its id list),
            # decoding is unambiguous and complete: return it immediately
            # without waiting for the loop to naturally terminate.
            # In the example this fires once "fn_add_numbers" is the only
            # name left AND all of its tokens have been committed.
            (only_name, only_ids), = candidates.items()
            if pos >= len(only_ids):
                return only_name
    # --- loop exited via `break` (allowed was empty) ---
    if not candidates:
        # Defensive fallback (should not normally trigger, since `next_id`
        # is always drawn from some candidate's expected token): if every
        # candidate got filtered out, fall back to scoring each full
        # candidate name by summing the logits its tokens would have
        # received at each step, and return the highest-scoring one. This
        # guarantees the function always returns a real name rather than
        # crashing, per the "no unhandled exceptions" requirement.
        logits = llm.get_logits_from_input_ids(input_ids)
        best = max(
            name_to_ids,
            key=lambda n: sum(logits[tid] for tid in name_to_ids[n] if tid < len(logits)),
        )
        if verbose:
            print(f"  [fn recovery] no candidates left, scored all -> {best}")
        return best

    # One or more candidates remain but all had their tokens fully consumed
    # at the same step (a tie in length with no single-survivor moment
    # above) — arbitrarily return the first one. In practice, with distinct
    # function names, this line is a safety net rather than the normal path.
    return next(iter(candidates))
