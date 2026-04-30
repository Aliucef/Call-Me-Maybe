NEG_INF = -1e30


def mask_logits_in_place(logits: list[float], allowed: set[int]) -> None:
    for i in range(len(logits)):
        if i not in allowed:
            logits[i] = NEG_INF


def allowed_next_number_tokens(id_to_token: list[str], built: str) -> set[int]:
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
    return allowed


def decode_number(
    llm,
    prompt_text: str,
    *,
    id_to_token: list[str],
    token_to_id: dict[str, int],
    param_name: str | None = None,
    max_steps: int = 12,
) -> float:
    tok = llm._tokenizer
    seed = tok.bos_token_id or tok.eos_token_id or tok.pad_token_id
    if seed is None:
        raise RuntimeError("Tokenizer has no BOS/EOS/PAD token id to seed decoding")

    if param_name:
        instr = f'Extract the number for parameter "{param_name}".'
    else:
        instr = "Extract the number."

    context = (
        f"{instr}\n"
        f"Request: {prompt_text}\n"
        "Answer with only the number:\n"
    )

    input_ids = [seed] + tok.encode(context, add_special_tokens=False)

    built = ""
    for _ in range(max_steps):
        logits = llm.get_logits_from_input_ids(input_ids)

        allowed = allowed_next_number_tokens(id_to_token, built=built)
        if not allowed:
            break

        mask_logits_in_place(logits, allowed)
        next_id = max(range(len(logits)), key=lambda i: logits[i])

        piece = id_to_token[next_id]
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
