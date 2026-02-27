from parser.parse import parse_args
from parser.validator import load_function_definitions, load_prompt_examples
from model.input_format import Output, OutputDict
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llm_sdk import Small_LLM_Model

def choose_function_name(llm, prompt_text: str, function_names: list[str]) -> str:
    tok = llm._tokenizer

    name_to_ids = {name: tok.encode(name, add_special_tokens=False) for name in function_names}
    name_to_ids = {n: ids for n, ids in name_to_ids.items() if ids}
    if not name_to_ids:
        raise RuntimeError("No valid tokenized function names")

    bos = tok.bos_token_id
    eos = tok.eos_token_id
    seed = bos if bos is not None else eos

    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    input_ids = [seed] + prompt_ids

    candidates = dict(name_to_ids)
    pos = 0

    while True:
        allowed = {ids[pos] for ids in candidates.values() if pos < len(ids)}
        if not allowed:
            break

        logits = llm.get_logits_from_input_ids(input_ids)
        next_id = max(allowed, key=lambda tid: logits[tid])

        input_ids.append(next_id)

        candidates = {
            name: ids for name, ids in candidates.items()
            if pos < len(ids) and ids[pos] == next_id
        }
        pos += 1

        if len(candidates) == 1:
            (only_name, only_ids), = candidates.items()
            if pos >= len(only_ids):
                return only_name

    # fallback (should be rare)
    return function_names[0]


def main():
    llm = Small_LLM_Model()
    args = parse_args()

    functions = load_function_definitions(args.functions_definition)
    promts = load_prompt_examples(args.input)

    for fn in functions:
        print("\n",fn.name )
        for pname, pschema in fn.parameters.items():
            print(pname, pschema.type)

    fn_names = [f.name for f in functions]

    print("=" * 20)
    for promt in promts:
        print(promt.prompt)
    output = []
    for prompt in promts:
        name = choose_function_name(llm, prompt.prompt, fn_names)
        obj = OutputDict(
            prompt= prompt.prompt,
            name= name,
            parameters={}
        )
        output.append(obj)


    text = json.dumps(
        [obj.model_dump() for obj in output],
        indent=2,
        ensure_ascii=False
    )
    args.output.write_text(text, encoding="utf-8")
    print(obj)
    print("+"* 40)
if __name__ == "__main__":
    main()
