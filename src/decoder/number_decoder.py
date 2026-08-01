from src.model.llm_protocol import LLMProtocol

NEG_INF = -1e30

# Cache allowed token sets per partial number string — the vocab is fixed per run.
_allowed_number_cache: dict[str, set[int]] = {}


def mask_logits_in_place(logits: list[float], allowed: set[int]) -> None:
    """Set logits of all tokens not in allowed to NEG_INF in place."""
    for i in range(len(logits)):
        if i not in allowed:
            logits[i] = NEG_INF


def allowed_next_number_tokens(id_to_token: list[str], built: str) -> set[int]:
    """Return the set of token IDs that keep the partial string a valid number prefix."""
    if built in _allowed_number_cache:
        return _allowed_number_cache[built]

    allowed: set[int] = set()
    for tid, tok in enumerate(id_to_token):
        if not tok:
            continue
        cand = built + tok
        s = cand.lstrip()
        if not s:
            continue
        if not all(ch.isdigit() or ch in "-." for ch in s):
            continue
        if s.count("-") > 1:
            continue
        if "-" in s and not s.startswith("-"):
            continue
        if s.count(".") > 1:
            continue
        t = s[1:] if s.startswith("-") else s
        if t.startswith("."):
            continue
        if len(t) >= 2 and t[0] == "0" and t[1].isdigit():
            continue
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
    """
    instr = (
        f'Extract the number for parameter "{param_name}".'
        if param_name else "Extract the number."
    )
    prior = ""
    if already_extracted:
        pairs = ", ".join(f'{k}={v}' for k, v in already_extracted.items())
        prior = f"Already extracted: {pairs}\n"
    context = (
        f"{instr}\n"
        f"{prior}"
        f"Request: {prompt_text}\n"
        "Answer with only the number:\n"
    )

    input_ids: list[int] = llm.encode(context).squeeze(0).tolist()
    built = ""

    for step in range(max_steps):
        logits = llm.get_logits_from_input_ids(input_ids)

        # Stop early when the model's top unconstrained choice is a stop token
        # (BPE newline Ċ, literal \n, or empty) — number is complete.
        best_overall = max(range(len(logits)), key=lambda i: logits[i])
        best_tok = id_to_token[best_overall]
        if not best_tok or "Ċ" in best_tok or "\n" in best_tok:
            break

        allowed = allowed_next_number_tokens(id_to_token, built=built)
        if not allowed:
            break

        mask_logits_in_place(logits, allowed)
        next_id = max(range(len(logits)), key=lambda i: logits[i])
        piece = id_to_token[next_id]

        if verbose:
            print(f"  [num step {step}] built={repr(built + piece)}")

        built += piece
        input_ids.append(next_id)

    s = built.strip()
    if s in {"", "-"}:
        return 0.0
    if s.endswith("."):
        s = s[:-1]
    try:
        return float(s)
    except ValueError:
        return 0.0
