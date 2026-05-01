*This project has been created as part of the 42 curriculum by Aliuc.*

# Call Me Maybe

## Description
This project is a function-calling tool.

It reads:
- a list of natural-language prompts
- a list of available function definitions

It writes a JSON file that matches each prompt with the most suitable function name and its arguments.

## Instructions
1. Install dependencies with `uv sync`.
2. Run the project with:

	`uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json`

3. Check the output JSON file in `data/output/`.

## Resources
- The project subject on function calling and constrained decoding
- Pydantic documentation: https://docs.pydantic.dev/
- Python JSON documentation: https://docs.python.org/3/library/json.html

AI was used to help summarize the subject and improve the wording of this README.

## Example usage
```bash
uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json
```

## Testing
Use the provided input files and verify that the output file is valid JSON and has the expected structure.
