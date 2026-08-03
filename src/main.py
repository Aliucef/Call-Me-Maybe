import json
import os
import sys
from pathlib import Path

from llm_sdk import Small_LLM_Model

from src.decoder.function_chooser import choose_function_name
from src.decoder.number_decoder import decode_number
from src.decoder.string_decoder import decode_string
from src.exceptions import InputFileError
from src.model.input_format import FunctionDef, OutputDict, PormptExample
from src.model.llm_protocol import LLMProtocol
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


def setup_llm(
    model: str,
    functions: list[FunctionDef],
) -> tuple[LLMProtocol, dict[str, list[int]], list[str]]:
    """Instantiate the LLM and build token lookup structures."""
    llm: LLMProtocol = Small_LLM_Model(model_name=model)
    name_to_ids: dict[str, list[int]] = {
        f.name: llm.encode(f.name).squeeze(0).tolist()
        for f in functions
    }
    name_to_ids = {n: ids for n, ids in name_to_ids.items() if ids}
    vocab_path = str(Path(model) / "vocab.json") if os.path.isdir(model) \
        else llm.get_path_to_vocab_file()
    _, id_to_token = load_vocab_maps(vocab_path)
    return llm, name_to_ids, id_to_token


def process_prompt(
    llm: LLMProtocol,
    item: PormptExample,
    functions: list[FunctionDef],
    fn_by_name: dict[str, FunctionDef],
    name_to_ids: dict[str, list[int]],
    id_to_token: list[str],
    verbose: bool,
) -> OutputDict:
    """Choose a function and decode all its parameters for one prompt."""
    if verbose:
        print(f"\nPrompt: {item.prompt}")
    name = choose_function_name(
        llm,
        prompt_text=item.prompt,
        functions=functions,
        name_to_ids=name_to_ids,
        verbose=verbose,
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
                verbose=verbose,
            )
        elif schema.type == "string":
            parameters[param_name] = decode_string(
                llm,
                prompt_text=item.prompt,
                id_to_token=id_to_token,
                param_name=param_name,
                already_extracted=dict(parameters),
                verbose=verbose,
            )
    print(f"{item.prompt}  ->  {name}  params={parameters}")
    return OutputDict(prompt=item.prompt, name=name, parameters=parameters)


def write_output(output: list[OutputDict], path: Path) -> None:
    """Serialize results to JSON and write to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([o.model_dump() for o in output], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    """Run the function-calling pipeline."""
    args = parse_args()
    try:
        functions = load_function_definitions(args.functions_definition)
        prompts = load_prompt_examples(args.input)
    except InputFileError as e:
        sys.exit(f"[ERROR] {e}")
    fn_by_name = {f.name: f for f in functions}
    llm, name_to_ids, id_to_token = setup_llm(args.model, functions)
    output = [
        process_prompt(llm, item, functions, fn_by_name, name_to_ids, id_to_token, args.verbose)
        for item in prompts
    ]
    write_output(output, args.output)


if __name__ == "__main__":
    main()
