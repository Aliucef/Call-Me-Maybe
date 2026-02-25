from parser.parse import parse_args
from parser.validator import load_function_definitions, load_prompt_examples
from model.input_format import Output, OutputDict
import json
def main():
    args = parse_args()

    functions = load_function_definitions(args.functions_definition)
    promts = load_prompt_examples(args.input)

    for fn in functions:
        print("\n",fn.name )
        for pname, pschema in fn.parameters.items():
            print(pname, pschema.type)
    default_name = functions[0].name

    print("=" * 20)
    for promt in promts:
        print(promt.prompt)
    output = []
    for prompt in promts:
        obj = OutputDict(
            prompt= prompt.prompt,
            name= default_name,
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
    print(obj)

main()
