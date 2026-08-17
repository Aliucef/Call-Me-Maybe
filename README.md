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
- **Public SDK surface only.** All LLM interaction goes through the public `LLMProtocol` surface (`encode()`, `decode()`, `get_logits_from_input_ids()`, `get_path_to_vocab_file()`) — no private `Small_LLM_Model` attributes are touched anywhere in `src/`. In practice, the current grounded-candidate design only ever calls `encode()` and `get_logits_from_input_ids()`; `decode()` and `get_path_to_vocab_file()` were needed by an earlier free-generation approach and are kept available on the protocol but currently unused.

## Performance analysis
- Model load (weight loading, ~311 shards) takes under a second on this machine; after that, every decoding step is a single forward pass.
- Function-name selection typically resolves in a handful of forward passes since the candidate set shrinks fast with each committed token; number/string candidate selection is bounded by the (usually short) length of the chosen literal.
- Measured end-to-end: the full bundled test set (11 prompts) completes in ~65 seconds wall-clock on this machine (CPU-only, float32, sandboxed), including model load — comfortably inside the 5-minute budget required by the subject. A single prompt (e.g. "What is the sum of 2 and 3?") resolves in well under a second of pure decoding time once the model is loaded.
- Because every step masks out anything that isn't a valid candidate continuation, output JSON is always syntactically valid and schema-compliant — the accuracy ceiling is governed only by whether the correct value is actually extractable from the prompt text, not by JSON malformation.

## Challenges faced
- An earlier version free-generated numbers/strings digit-by-digit/token-by-token from the whole vocabulary; it occasionally produced values with no basis in the prompt. Replacing that with prompt-grounded candidate narrowing removed the hallucination risk entirely.
- The subject forbids private `llm_sdk` attributes, so every token id used for masking has to come from the public `encode()` method (tokenizing whole candidate strings) rather than walking the tokenizer's private vocabulary object directly. An earlier free-generation design did need to read the raw vocab file via `get_path_to_vocab_file()` to build a full-vocabulary digit/character mask; grounded-candidate encoding replaced that need entirely.
- Handling repeated numeric literals correctly (e.g. distinguishing "sum of 2 and 2" from "sum of 2 and ?") required counting occurrences of each value rather than just checking set membership.
- The shared candidate-narrowing loop (`choose_from_candidates()`) cannot stop on a candidate that's an exact token-prefix of another (e.g. "cat" vs "cats") — it always keeps extending toward the longer one. Not currently reachable through the bundled functions/prompts, but documented and pinned by a test (`tests/test_candidate_chooser.py::test_shared_prefix_resolves_to_the_longer_candidate`) so it isn't rediscovered blind.

## Testing strategy
- `tests/` contains 39 `pytest` unit tests across five files — `test_function_chooser.py`, `test_number_decoder.py`, `test_string_decoder.py`, `test_boolean_decoder.py`, and `test_candidate_chooser.py` — exercising every decoder plus the shared `choose_from_candidates()` primitive against a shared mocked LLM (`tests/conftest.py`), so core candidate-narrowing logic can be verified without downloading the model.
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

AI was used to: help summarize the project subject into actionable requirements, review the codebase against those requirements to find gaps (private-attribute usage, dead code, missing `.gitignore` rules), help word this README, add explanatory inline comments throughout `src/` for defense preparation, and — after AI review found that a prior cleanup commit had silently dropped the caching optimization and its test coverage while `BONUSES.md` still claimed both — restore an equivalent cache (`src/decoder/number_decoder.py:encode_number_candidate()`) and write the test suite in `tests/` (39 tests across 5 files) so those bonus claims are true again. All AI-suggested and AI-written changes were reviewed by the author, who is responsible for understanding and defending every part of this codebase, including the AI-assisted portions, during evaluation.
