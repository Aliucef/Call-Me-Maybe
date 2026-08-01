# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
make install        # or: uv sync

# Run the pipeline
make run            # reads from data/input/, writes to data/output/

# Manual run with explicit paths
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json

# Lint (flake8 + mypy with standard flags)
make lint

# Strict mypy
make lint-strict

# Clean pycache
make clean
```

There is no test suite. Correctness is verified by inspecting `data/output/function_calling_results.json`.

## Subject constraints (mandatory)

These rules come directly from the project subject and are graded:

- **Only public SDK methods may be used** — calling `llm._tokenizer` or any other private attribute/method of `Small_LLM_Model` is explicitly forbidden. The only allowed public API is:
  - `get_logits_from_input_ids(input_ids: List[int]) -> List[float]`
  - `get_path_to_vocab_file() -> str`
  - `encode(text: str) -> Tensor`
  - `decode(token_ids: List[int]) -> str` (optional)
- **No `dspy`, `torch`, `transformers`, `huggingface_hub`** in student code — these live inside `llm_sdk/` (provided), but the student's `src/` must not import them directly. Only `pydantic`, `numpy`, and `json` are allowed as student dependencies.
- **Function selection must use the LLM** — heuristic/keyword matching is forbidden.
- **Input files may change** during peer review — never hardcode function names or prompts.
- **Do not commit `data/output/`** — it is generated at runtime.
- **Code style**: flake8-compliant, full type hints, mypy must pass without errors, docstrings required (PEP 257).
- **No unhandled exceptions** — the program must never crash unexpectedly.

### Known violation in current code

`src/decoder/function_chooser.py` and `src/main.py` both access `llm._tokenizer` directly (a private attribute). This must be replaced with the public `encode()` method or the vocab file approach.

### Missing feature

String parameters (`"type": "string"`) have no constrained decoder — `src/main.py` only decodes `"number"` typed params. A string decoder is needed for full compliance.

## Architecture

This is a 42-school project that performs **LLM-based function calling via constrained decoding**. Given a set of natural-language prompts and a JSON schema of available functions, it outputs a JSON file matching each prompt to a function name and its arguments.

### Data flow

```
data/input/functions_definition.json  ─┐
data/input/function_calling_tests.json ─┤─> src/main.py ─> data/output/function_calling_results.json
```

### Key modules

**`llm_sdk/`** — Wrapper around a HuggingFace causal LM (`Qwen/Qwen3-0.6B` by default). `Small_LLM_Model` loads the model, exposes `get_logits_from_input_ids()` for raw next-token logits, and provides helpers to download vocab/tokenizer files. Device selection priority: MPS > CUDA > CPU.

**`src/model/input_format.py`** — Pydantic models: `FunctionDef` (name, description, parameters schema), `PromptExample` (prompt string), `OutputDict` (prompt + chosen name + extracted parameters).

**`src/parser/`** — `parse.py` handles CLI args (`--functions_definition`, `--input`, `--output`). `validator.py` loads and validates the two JSON input files against the Pydantic models.

**`src/decoder/function_chooser.py`** — Constrained decoding for function selection. Builds a prompt listing all available functions, then greedily decodes the function name token-by-token. At each step, only tokens that are a valid continuation of at least one candidate function name are allowed (all others get logit −∞). Eliminates candidates as tokens are committed until one name remains.

**`src/decoder/number_decoder.py`** — Constrained decoding for numeric parameters. At each step, `allowed_next_number_tokens()` filters the vocabulary to only tokens that keep the partially-built string a valid number literal (handles sign, decimal point, leading-zero rules). Decodes up to `max_steps=12` tokens then parses with `float()`.

**`src/pipeline/pipeline.py`** — Contains `heuristic_choose()`, a keyword-based fallback for function selection (not used in the main path; kept as reference).

**`src/main.py`** — Orchestrates the full pipeline: load inputs → instantiate `Small_LLM_Model` → for each prompt, call `choose_function_name` then `decode_number` for each `"number"` typed parameter → serialize output JSON.

### Constrained decoding pattern

The core technique: instead of free-form generation, logits are masked so only structurally valid tokens score above −∞ at each step. `function_chooser.py` and `number_decoder.py` both follow this pattern — build a candidate set, call `get_logits_from_input_ids`, pick the best allowed token, append it to `input_ids`, narrow the candidate set, repeat.

### Input/output format

`functions_definition.json` — array of objects with `name`, `description`, `parameters` (dict of `{param_name: {type: "number"|"string"}}`), optional `returns`.

`function_calling_tests.json` — array of `{"prompt": "..."}` objects.

Output — array of `{"prompt": ..., "name": ..., "parameters": {...}}` objects.
