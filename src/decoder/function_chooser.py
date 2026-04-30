from llm_sdk import Small_LLM_Model


def choose_function_name(
    llm: Small_LLM_Model,
    prompt_text: str,
    functions,
    name_to_ids: dict[str, list[int]],
) -> str:
    tok = llm._tokenizer
    bos = tok.bos_token_id
    eos = tok.eos_token_id
    seed = bos if bos is not None else eos

    lines = []
    for f in functions:
        line = f"- {f.name}: {f.description or ''} params={list(f.parameters.keys())}"
        lines.append(line)

    menu = "\n".join(lines)
    context = (
        "Choose the best function for the request.\n"
        "Available functions:\n"
        f"{menu}\n"
        f"Request: {prompt_text}\n"
        "Answer with the function name only:\n"
    )

    input_ids = [seed] + tok.encode(context, add_special_tokens=False)
    candidates = dict(name_to_ids)
    pos = 0

    while True:
        allowed = {ids[pos] for ids in candidates.values() if pos < len(ids)}
        if not allowed:
            break

        logits = llm.get_logits_from_input_ids(input_ids)
        next_id = max(allowed, key=lambda tid: logits[tid])

        input_ids.append(next_id)

        candidates = {
            name: ids
            for name, ids in candidates.items()
            if pos < len(ids) and ids[pos] == next_id
        }
        pos += 1

        if len(candidates) == 1:
            (only_name, only_ids), = candidates.items()
            if pos >= len(only_ids):
                return only_name
    print("hello")
    return next(iter(name_to_ids.keys()))
