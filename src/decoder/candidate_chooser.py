from src.model.llm_protocol import LLMProtocol


def choose_from_candidates(
    llm: LLMProtocol,
    context: str,
    candidates_by_key: dict[str, list[int]],
    verbose: bool = False,
    log_label: str = "choice",
) -> str:
    """Select the best matching candidate using constrained greedy decoding."""
    input_ids: list[int] = llm.encode(context).squeeze(0).tolist()
    original = dict(candidates_by_key)
    candidates = dict(candidates_by_key)
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
                key=lambda k: logits[candidates[k][pos]] if pos < len(candidates[k]) else -1e30,
                reverse=True,
            )[:3]
            print(f"  [{log_label} step {pos}] candidates={top} -> token_id={next_id}")

        input_ids.append(next_id)
        candidates = {
            key: ids
            for key, ids in candidates.items()
            if pos < len(ids) and ids[pos] == next_id
        }
        pos += 1

        if len(candidates) == 1:
            (only_key, only_ids), = candidates.items()
            if pos >= len(only_ids):
                return only_key

    if not candidates:
        logits = llm.get_logits_from_input_ids(input_ids)
        best = max(
            original,
            key=lambda k: sum(logits[tid] for tid in original[k] if tid < len(logits)),
        )
        if verbose:
            print(f"  [{log_label} recovery] no candidates left, scored all -> {best}")
        return best

    return next(iter(candidates))
