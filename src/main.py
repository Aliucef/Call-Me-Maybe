import json
import os
from pathlib import Path

from llm_sdk import Small_LLM_Model

from src.decoder.function_chooser import choose_function_name
from src.decoder.number_decoder import decode_number
from src.decoder.string_decoder import decode_string
from src.model.input_format import OutputDict
from src.parser.parse import parse_args
from src.parser.validator import load_function_definitions, load_prompt_examples


def load_vocab_maps(vocab_path: str) -> tuple[dict[str, int], list[str]]:
    """Load vocab.json and build a reverse id-to-token list."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        token_to_id: dict[str, int] = json.load(f)
    vocab_size = len(token_to_id)
    id_to_token = [""] * vocab_size
    for tok, tid in token_to_id.items():
        if 0 <= tid < vocab_size:
            id_to_token[tid] = tok
    return token_to_id, id_to_token


def main() -> None:
    """Run the function-calling pipeline."""
    args = parse_args()

    functions = load_function_definitions(args.functions_definition)
    prompts = load_prompt_examples(args.input)

    fn_by_name = {f.name: f for f in functions}

    llm = Small_LLM_Model(model_name=args.model)

    # Encode each function name using the public API
    name_to_ids: dict[str, list[int]] = {
        f.name: llm.encode(f.name).squeeze(0).tolist()
        for f in functions
    }
    name_to_ids = {n: ids for n, ids in name_to_ids.items() if ids}

    # For local model directories, find vocab.json directly instead of going
    # through hf_hub_download (which only understands Hub repo IDs).
    if os.path.isdir(args.model):
        vocab_path = str(Path(args.model) / "vocab.json")
    else:
        vocab_path = llm.get_path_to_vocab_file()
    _, id_to_token = load_vocab_maps(vocab_path)

    output: list[OutputDict] = []

    for item in prompts:
        if args.verbose:
            print(f"\nPrompt: {item.prompt}")

        name = choose_function_name(
            llm,
            prompt_text=item.prompt,
            functions=functions,
            name_to_ids=name_to_ids,
            verbose=args.verbose,
        )

        fn = fn_by_name[name]
        parameters: dict[str, object] = {}

        for param_name, schema in fn.parameters.items():
            if schema.type == "number":
                parameters[param_name] = decode_number(
                    llm,
                    prompt_text=item.prompt,
                    id_to_token=id_to_token,
                    param_name=param_name,
                    already_extracted=dict(parameters),
                    verbose=args.verbose,
                )
            elif schema.type == "string":
                parameters[param_name] = decode_string(
                    llm,
                    prompt_text=item.prompt,
                    id_to_token=id_to_token,
                    param_name=param_name,
                    already_extracted=dict(parameters),
                    verbose=args.verbose,
                )

        output.append(OutputDict(prompt=item.prompt, name=name, parameters=parameters))
        print(f"{item.prompt}  ->  {name}  params={parameters}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps([o.model_dump() for o in output], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
