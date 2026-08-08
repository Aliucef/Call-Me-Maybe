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
    # vocab.json (downloaded via llm.get_path_to_vocab_file()) is a plain
    # JSON object mapping token text -> integer id, e.g. {"fn": 16, "_": 62,
    # "add": 1997, ..., "Ċ": 198, ...} — the same format HuggingFace
    # tokenizers ship for GPT-2/BPE-style vocabularies.
    with open(vocab_path, "r", encoding="utf-8") as f:
        token_to_id: dict[str, int] = json.load(f)
    vocab_size = len(token_to_id)
    # id_to_token is the inverse lookup, as a list indexed by token id —
    # this is what every decoder (function_chooser, number_decoder,
    # string_decoder) uses to turn a chosen id back into readable text
    # (e.g. to check "does this token contain a newline?") without ever
    # calling a private tokenizer method.
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
    # Loads the HuggingFace model wrapped by Small_LLM_Model, e.g.
    # "Qwen/Qwen3-0.6B" (the default from src/utils/constants.py).
    llm: LLMProtocol = Small_LLM_Model(model_name=model)
    # Pre-tokenize every function name once, up front, so
    # choose_function_name() doesn't need to call the (public) encode()
    # method repeatedly per prompt. Example:
    #   {"fn_add_numbers": [16, 1997, ...], "fn_greet": [16, 6820, ...], ...}
    # llm.encode(name) returns a 2-D tensor shaped [1, num_tokens];
    # .squeeze(0) drops the batch dimension, .tolist() converts it to a
    # plain Python list[int] (student code never touches torch directly).
    name_to_ids: dict[str, list[int]] = {
        f.name: llm.encode(f.name).squeeze(0).tolist()
        for f in functions
    }
    # Defensive: drop any name that somehow tokenized to nothing.
    name_to_ids = {n: ids for n, ids in name_to_ids.items() if ids}
    # Locate vocab.json: if `model` is a local directory (e.g. during
    # offline testing) read vocab.json from inside it directly; otherwise
    # ask the SDK's public get_path_to_vocab_file(), which downloads it
    # from the HuggingFace Hub and returns a local cache path.
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
    # RUNNING EXAMPLE: item.prompt = "What is the sum of 2 and 3?"
    if verbose:
        print(f"\nPrompt: {item.prompt}")
    # Step 1: pick which function this prompt is asking to call.
    # See src/decoder/function_chooser.py for the full constrained-decoding
    # walkthrough. Result here: name == "fn_add_numbers".
    name = choose_function_name(
        llm,
        prompt_text=item.prompt,
        functions=functions,
        name_to_ids=name_to_ids,
        verbose=verbose,
    )
    # Look up the full FunctionDef (with its parameter schema) by name.
    fn = fn_by_name[name]
    parameters: dict[str, object] = {}
    # Step 2: decode each parameter in the order it appears in the schema.
    # For fn_add_numbers this is {"a": number, "b": number}, so this loop
    # calls decode_number twice: once for "a", once for "b".
    for param_name, schema in fn.parameters.items():
        if schema.type == "number":
            # dict(parameters) passes a snapshot of what's been decoded so
            # far (e.g. {"a": 2.0} on the second iteration) so the decoder
            # can tell the model what's already been extracted — see the
            # `already_extracted` docstring note in number_decoder.py.
            parameters[param_name] = decode_number(
                llm,
                prompt_text=item.prompt,
                id_to_token=id_to_token,
                param_name=param_name,
                already_extracted=dict(parameters),
                verbose=verbose,
            )
        elif schema.type == "string":
            # Same pattern for string-typed parameters, e.g. fn_greet's
            # "name" parameter on the prompt "Greet shrek".
            parameters[param_name] = decode_string(
                llm,
                prompt_text=item.prompt,
                id_to_token=id_to_token,
                param_name=param_name,
                already_extracted=dict(parameters),
                verbose=verbose,
            )
        # Any other schema.type is silently skipped — the subject only
        # defines "number" and "string" parameter types.
    # Example final line printed to stdout:
    #   "What is the sum of 2 and 3?  ->  fn_add_numbers  params={'a': 2.0, 'b': 3.0}"
    print(f"{item.prompt}  ->  {name}  params={parameters}")
    return OutputDict(prompt=item.prompt, name=name, parameters=parameters)


def write_output(output: list[OutputDict], path: Path) -> None:
    """Serialize results to JSON and write to disk."""
    # Ensure data/output/ (or whatever --output's parent dir is) exists;
    # exist_ok=True means this is a no-op if it's already there.
    path.parent.mkdir(parents=True, exist_ok=True)
    # o.model_dump() turns each pydantic OutputDict back into a plain dict
    # (e.g. {"prompt": ..., "name": ..., "parameters": {...}}), and
    # json.dumps serializes the whole list with 2-space indentation.
    # ensure_ascii=False keeps non-ASCII characters (e.g. accented names)
    # readable in the output file instead of \uXXXX-escaped.
    path.write_text(
        json.dumps([o.model_dump() for o in output], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    """Run the function-calling pipeline."""
    # Parses --functions_definition, --input, --output, --model, --verbose.
    args = parse_args()
    try:
        # Load + pydantic-validate both input JSON files. Example:
        # functions == [FunctionDef(name="fn_add_numbers", ...), ...]
        # prompts   == [PormptExample(prompt="What is the sum of 2 and 3?"), ...]
        functions = load_function_definitions(args.functions_definition)
        prompts = load_prompt_examples(args.input)
    except InputFileError as e:
        # Any of the four custom exception types from src/exceptions/ lands
        # here — printed as a clean one-line error and the process exits
        # with a non-zero status, instead of a raw Python traceback. This
        # is the "no unhandled exceptions" requirement in practice.
        sys.exit(f"[ERROR] {e}")
    # Quick lookup table so process_prompt() can go name -> FunctionDef in
    # O(1) instead of scanning the list each time.
    fn_by_name = {f.name: f for f in functions}
    # Load the model once and pre-tokenize every function name once —
    # reused across every prompt below.
    llm, name_to_ids, id_to_token = setup_llm(args.model, functions)
    # Process every prompt in the input file, one at a time, building the
    # final list of results. For the two examples used throughout these
    # comments, this produces entries like:
    #   {"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers",
    #    "parameters": {"a": 2.0, "b": 3.0}}
    #   {"prompt": "Greet shrek", "name": "fn_greet",
    #    "parameters": {"name": "shrek"}}
    output = [
        process_prompt(llm, item, functions, fn_by_name, name_to_ids, id_to_token, args.verbose)
        for item in prompts
    ]
    # Write the whole list to --output as one JSON array.
    write_output(output, args.output)


if __name__ == "__main__":
    # Entry point when run as `python -m src` (see src/__main__.py) or
    # directly as a script.
    main()
