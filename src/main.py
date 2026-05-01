from src.parser.parse import parse_args
from src.parser.validator import load_function_definitions, load_prompt_examples
from src.model.input_format import OutputDict
from src.decoder.number_decoder import decode_number

import json
from llm_sdk import Small_LLM_Model
from src.decoder.function_chooser import choose_function_name


def load_vocab_maps(vocab_path: str):
    with open(vocab_path, "r", encoding="utf-8") as f:
        token_to_id: dict[str, int] = json.load(f)

    vocab_size = len(token_to_id)
    id_to_token = [""] * vocab_size
    for tok, tid in token_to_id.items():
        if 0 <= tid < vocab_size:
            id_to_token[tid] = tok
    return token_to_id, id_to_token


def main():
    args = parse_args()

    functions = load_function_definitions(args.functions_definition)
    prompts = load_prompt_examples(args.input)

    fn_by_name = {f.name: f for f in functions}

    llm = Small_LLM_Model()
    tok = llm._tokenizer

    # function-name constraints
    name_to_ids = {f.name: tok.encode(f.name, add_special_tokens=False) for f in functions}
    name_to_ids = {n: ids for n, ids in name_to_ids.items() if ids}
    vocab_path = llm.get_path_to_vocab_file()
    token_to_id, id_to_token = load_vocab_maps(vocab_path)

    output: list[OutputDict] = []

    for item in prompts:
        name = choose_function_name(
            llm,
            prompt_text=item.prompt,
            functions=functions,
            name_to_ids=name_to_ids,
        )

        fn = fn_by_name[name]

        parameters = {}
        for param_name, schema in fn.parameters.items():
            if schema.type == "number":
                val = decode_number(
                    llm,
                    prompt_text=item.prompt,
                    id_to_token=id_to_token,
                    token_to_id=token_to_id,
                    param_name=param_name,   # if your decoder supports it; otherwise remove
                )
                parameters[param_name] = val

        output.append(
            OutputDict(
                prompt=item.prompt,
                name=name,
                parameters=parameters,
            )
        )

        print(f"{item.prompt}  ->  {name}  params={parameters}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps([o.model_dump() for o in output], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
