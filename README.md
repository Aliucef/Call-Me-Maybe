*This project has been created as part of the 42 curriculum by Aliuc.*

# Call Me Maybe

## Description
This project is a function-calling tool: it turns natural-language requests into structured, machine-executable function calls.

It reads:
- a list of natural-language prompts (`function_calling_tests.json`)
- a list of available function definitions (`functions_definition.json`)

For each prompt it writes a JSON object matching the prompt to the most suitable function name and its typed arguments — using a small local LLM (`Qwen/Qwen3-0.6B` by default) driven entirely through **constrained decoding**, never through keyword heuristics.

## Instructions
1. Install dependencies with `uv sync` (or `make install`).
2. Run the project with:

	`uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json`

	or simply `make run`.
3. Check the output JSON file in `data/output/`.
4. Other Makefile targets: `make debug` (runs under `pdb`), `make lint` / `make lint-strict` (flake8 + mypy), `make clean` (removes caches).

## Algorithm explanation
Every decision the model makes is constrained at the logit level: only token IDs that keep the output inside a known-valid set of candidates are ever eligible, so nothing outside that set can ever be produced.

- **Function selection** (`src/decoder/function_chooser.py`): a prompt listing all available functions (name, description, parameters) is built, then the function name is decoded greedily token-by-token. At each step, only tokens that continue at least one surviving candidate function name are allowed (`get_logits_from_input_ids` logits are masked with `-inf` everywhere else); candidates are eliminated as tokens commit until exactly one name remains.
- **Number parameters** (`src/decoder/number_decoder.py`): rather than letting the model free-generate digits (which a 0.6B model can hallucinate), the numeric literals that actually appear in the prompt are extracted with a regex, and the model picks between them using the same candidate-narrowing mechanism as function selection. Occurrence counting (not just distinct values) makes sure "sum of 2 and 2" still allows both parameters to resolve to `2.0`, while "sum of 2 and ?" correctly runs out of candidates for the second parameter instead of silently repeating the first.
- **String parameters** (`src/decoder/string_decoder.py`): candidates are gathered in three tiers, each only consulted if the previous one is empty — quoted substrings, a small built-in library of regex patterns for textual-category concepts mentioned in the prompt (e.g. "vowels" → `[aeiouAEIOU]`), and finally the prompt's trailing unquoted word as a last-resort positional guess. The model then picks among those candidates with the same constrained-decoding mechanism.
- **Shared core** (`src/decoder/candidate_chooser.py`): `choose_from_candidates()` implements the actual masked, candidate-narrowing decode loop used by all three of the above — it's the one place logits get masked and the one guarantee that an output can only ever be a value from the supplied candidate set.

## Design decisions
- **Grounded candidates over free generation.** Numbers and strings are always picked from values that literally appear in the prompt (or a small closed set of domain-knowledge regex patterns), instead of generating tokens freely. This trades a small amount of flexibility for the guarantee that the model can never fabricate a value with no basis in the request.
- **One shared decoding primitive.** Function-name selection and number/string candidate picking all funnel through the same `choose_from_candidates()`, keeping the masking logic in a single, well-tested place instead of duplicated across decoders.
- **Explicit failure over silent fabrication.** When no grounded candidate exists for a required parameter (e.g. the prompt genuinely omits it), the resolver prints a clear `[ERROR]` message and defaults to `0.0` / `""` rather than crashing or guessing — the output schema always requires the key to be present.
- **Public SDK surface only.** All LLM interaction goes through `encode()`, `decode()`, `get_logits_from_input_ids()`, and `get_path_to_vocab_file()` — no private `Small_LLM_Model` attributes are touched anywhere in `src/`.

## Performance analysis
- Model load (weight loading, ~311 shards) takes roughly a second on this machine; after that, every decoding step is a single forward pass.
- Function-name selection typically resolves in a handful of forward passes since the candidate set shrinks fast with each committed token; number/string candidate selection is bounded by the (usually short) length of the chosen literal.
- The full bundled test set (11 prompts) runs in a few seconds once the model is loaded, comfortably inside the 5-minute budget required by the subject.
- Because every step masks out anything that isn't a valid candidate continuation, output JSON is always syntactically valid and schema-compliant — the accuracy ceiling is governed only by whether the correct value is actually extractable from the prompt text, not by JSON malformation.

## Challenges faced
- An earlier version free-generated numbers/strings digit-by-digit/token-by-token from the whole vocabulary; it occasionally produced values with no basis in the prompt. Replacing that with prompt-grounded candidate narrowing removed the hallucination risk entirely.
- The subject forbids private `llm_sdk` attributes, so the token↔id vocabulary needed for masking has to come from the public `get_path_to_vocab_file()` / `encode()` / `decode()` methods rather than the tokenizer object directly.
- Handling repeated numeric literals correctly (e.g. distinguishing "sum of 2 and 2" from "sum of 2 and ?") required counting occurrences of each value rather than just checking set membership.

## Testing strategy
- `tests/` contains `pytest` unit tests (e.g. `test_function_chooser.py`) that exercise the decoders against a mocked LLM, so core candidate-narrowing logic can be verified without downloading the model.
- End-to-end verification runs the full pipeline against `data/input/` and inspects `data/output/function_calling_results.json` for schema-correct output, including edge cases such as prompts that omit a required value entirely (e.g. "Greet" with no name, "What is the sum of and ?").
- `make lint` and `make lint-strict` (flake8 + mypy, including `--strict`) are run on every change to keep the codebase type-safe and style-compliant.

## Example usage
```bash
uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calling_results.json
```

## Resources
- The project subject on function calling and constrained decoding
- Pydantic documentation: https://docs.pydantic.dev/
- Python JSON documentation: https://docs.python.org/3/library/json.html

AI was used to: help summarize the project subject into actionable requirements, review the codebase against those requirements to find gaps (private-attribute usage, dead code, missing `.gitignore` rules), and help word this README. All implementation code (decoders, constrained-decoding logic, pipeline) was written and understood by the author; AI-suggested changes were reviewed before being applied.
