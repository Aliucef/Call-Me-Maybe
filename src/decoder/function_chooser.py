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
    """
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

    input_ids: list[int] = llm.encode(context).squeeze(0).tolist()
    candidates = dict(name_to_ids)
    pos = 0

    while True:
        allowed = {ids[pos] for ids in candidates.values() if pos < len(ids)}
        if not allowed:
            break

        logits = llm.get_logits_from_input_ids(input_ids)
        next_id = max(allowed, key=lambda tid: logits[tid])

        if verbose:
            top = sorted(
                candidates,
                key=lambda n: logits[candidates[n][pos]] if pos < len(candidates[n]) else -1e30,
                reverse=True,
            )[:3]
            print(f"  [fn step {pos}] candidates={top} -> token_id={next_id}")

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

    # Recovery: score every function name by the sum of logits of its tokens
    if not candidates:
        logits = llm.get_logits_from_input_ids(input_ids)
        best = max(
            name_to_ids,
            key=lambda n: sum(logits[tid] for tid in name_to_ids[n] if tid < len(logits)),
        )
        if verbose:
            print(f"  [fn recovery] no candidates left, scored all -> {best}")
        return best

    return next(iter(candidates))
