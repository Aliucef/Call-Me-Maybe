from parser.parse import parse_args
from parser.validator import load_function_definitions, load_prompt_examples

def main():
    args = parse_args()

    functions = load_function_definitions(args.functions_definition)
    promts = load_prompt_examples(args.input)

    # for fn in functions:
    #     print("\n",fn.name )
    #     for pname, pschema in fn.parameters.items():
    #         print(pname, pschema.type)


    # print("=" * 20)
    # for promt in promts:
    #     print(promt.prompt)
main()