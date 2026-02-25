def heuristic_choose(prompt: str) -> str:
    p = prompt.lower()

    # order matters: more specific first
    if "square root" in p:
        return "fn_get_square_root"

    if "reverse" in p:
        return "fn_reverse_string"

    if "greet" in p or "hello" in p and "greet" in p:
        return "fn_greet"

    if "replace" in p or "substitute" in p or "regex" in p:
        return "fn_substitute_string_with_regex"

    if "sum" in p or "add" in p or "plus" in p:
        return "fn_add_numbers"

    # fallback
    return "fn_add_numbers"
