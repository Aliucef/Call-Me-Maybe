import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llm_sdk import Small_LLM_Model

NEG_INF = -1e30  # -infinity


def argmax(xs: list[float]) -> int:
    best_i = 0
    best_v = xs[0]
    for i, v in enumerate(xs):
        if v > best_v:
            best_v = v
            best_i = i
    return best_i


def load_vocab_maps(vocab_path: str):
    with open(vocab_path, "r", encoding="utf-8") as f:
        token_to_id: dict[str, int] = json.load(f)

    vocab_size = len(token_to_id)
    id_to_token = [""] * vocab_size
    for tok, tid in token_to_id.items():
        if 0 <= tid < vocab_size:
            id_to_token[tid] = tok
    return token_to_id, id_to_token


def mask_logits_in_place(logits: list[float], allowed_ids: set[int]) -> None:
    for i in range(len(logits)):
        if i not in allowed_ids:
            logits[i] = NEG_INF


def allowed_for_prefix(
    id_to_token: list[str],
    built: str,
    target: str,
) -> set[int]:
    """
    Return token IDs whose token string keeps (built + token) a prefix of target.
    """
    allowed: set[int] = set()
    for tid, tok in enumerate(id_to_token):
        if not tok:
            continue
        candidate = built + tok
        if len(candidate) > len(target):
            continue
        if target.startswith(candidate):
            allowed.add(tid)
    return allowed


def main():
    llm = Small_LLM_Model() # from ll_sdk qwen 0.6 b params

    vocab_path = llm.get_path_to_vocabulary_json() # get the path to the vocab.json file
    token_to_id, id_to_token = load_vocab_maps(vocab_path) # this will load the the id : token and token : id from the vocabs
    # usually it returns def load_vocab_maps(vocab_path: str) -> tuple[dict[str, int], list[str]]

    # IMPORTANT: model cannot accept empty input_ids. Seed with BOS (or EOS fallback).
    bos = llm._tokenizer.bos_token_id # google this
    eos = llm._tokenizer.eos_token_id
    seed_id = bos if bos is not None else eos
    input_ids: list[int] = [seed_id]

    LBRACE = token_to_id["{"] #This will print the ID of token {
    QUOTE = token_to_id['"'] #same here for "
    RBRACE = token_to_id["}"] # this will print the ID of token }


    # State 1: force "{"
    logits = llm.get_logits_from_input_ids(input_ids) # scores for each token // still gotta figure out whats input_ids
    mask_logits_in_place(logits, {LBRACE}) # this will mask everything until its {
    next_id = argmax(logits)
    input_ids.append(next_id)

    # State 2: force '"'
    logits = llm.get_logits_from_input_ids(input_ids) # same here
    mask_logits_in_place(logits, {QUOTE})
    next_id = argmax(logits)
    input_ids.append(next_id)

    # State 3: generate key content = "prompt" using prefix constraints
    target_key = "prompt"
    built = ""

    logits = llm.get_logits_from_input_ids(input_ids)
    mask_logits_in_place(logits, {RBRACE})
    next_id = argmax(logits)
    input_ids.append( next_id)

    while built != target_key:
        logits = llm.get_logits_from_input_ids(input_ids)

        allowed = allowed_for_prefix(id_to_token, built=built, target=target_key)
        if not allowed:
            raise RuntimeError(
                f"No allowed tokens to continue building {target_key!r} from {built!r}"
            )

        mask_logits_in_place(logits, allowed)
        next_id = argmax(logits)
        tok = id_to_token[next_id]

        built += tok
        input_ids.append(next_id)

    # After key is complete, force closing quote
    logits = llm.get_logits_from_input_ids(input_ids)
    mask_logits_in_place(logits, {QUOTE})
    next_id = argmax(logits)
    input_ids.append(next_id)

    # Show result (skip the seed token for display)
    out_text = llm._tokenizer.decode(input_ids[1:], skip_special_tokens=True)
    print("Generated:", repr(out_text))


if __name__ == "__main__":
    main()
