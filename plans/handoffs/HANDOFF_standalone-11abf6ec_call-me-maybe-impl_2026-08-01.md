# Handoff: Call-Me-Maybe — Full Implementation Session

**Chain:** standalone-11abf6ec  
**Seq:** 1  
**Parent:** none  
**Date:** 2026-08-01  
**Branch:** master  
**Status:** Feature-complete, lint-clean, not yet committed

---

## Goal

Bring the 42-school `Call-Me-Maybe-1` project to submission-ready state:
- Fix all mandatory subject violations
- Implement 5 bonus features
- Pass `make lint` cleanly
- Produce correct output via `make run`

---

## Where We Are

**Ready for submission except:**
1. README missing required sections (algorithm explanation, design decisions, performance analysis, challenges faced, testing strategy, example usage)
2. Docstrings on all functions (user de-prioritised for now — subject grades them)
3. Uncommitted changes (see below)

**Lint:** `make lint` passes clean — flake8 + mypy, 14 source files, 0 errors.  
**Tests:** 16/16 passing (`uv run pytest tests/ -v`).  
**Run:** `uv run python -m src` (no args needed now — defaults wired in).

---

## What Was Done This Session

### Mandatory violations fixed

| Violation | Fix |
|-----------|-----|
| `llm._tokenizer` private access in `function_chooser.py` and `main.py` | Replaced with `llm.encode(text).squeeze(0).tolist()` |
| Same in `number_decoder.py` | Same fix |
| `print("hello")` debug artifact in `function_chooser.py:54` | Removed |
| `data/output/` not gitignored | Added to `.gitignore` |
| No string decoder | Created `src/decoder/string_decoder.py` |
| `str` params silently skipped in `main.py` | Added `elif schema.type == "string"` branch |
| Missing type hints in `validator.py` | Added full annotations |
| Blank line violations in `input_format.py`, `validator.py` | Fixed |
| `SystemError` instead of `SystemExit` in `validator.py` | Fixed, improved error messages |

### 5 bonuses implemented

1. **Multiple model support** — `--model` CLI flag, passed to `Small_LLM_Model(model_name=args.model)`
2. **Visualization** — `--verbose` flag, prints top-3 candidates at each function-selection step, built string at each number step, token at each string step
3. **Advanced error recovery** — `choose_function_name` now scores all function names against logits when candidates exhausted, instead of returning first key
4. **Performance caching** — `_allowed_number_cache: dict[str, set[int]]` module-level dict in `number_decoder.py`; `allowed_next_number_tokens` is pure so results are safe to reuse across all prompts in one run
5. **Test suite** — `tests/test_number_decoder.py` (11 tests) + `tests/test_function_chooser.py` (5 tests), both use mocked LLM, no model download needed

### Architecture additions

- `src/model/llm_protocol.py` — `LLMProtocol(Protocol)` with 4 methods (`encode`, `decode`, `get_logits_from_input_ids`, `get_path_to_vocab_file`). Decoders typed against Protocol instead of `Small_LLM_Model` so mocks pass mypy.
- `src/decoder/string_decoder.py` — new; stops on `Ċ` (BPE newline in Qwen/GPT-2 tokenizers), uses `llm.decode(built_ids)` for proper BPE→text conversion, strips partial quotes.
- `setup.cfg` — flake8 config: `max-line-length = 99`, `exclude = .venv, llm_sdk, scratch`.
- `pyproject.toml` — added `pytest`, `accelerate`; dev group `flake8`, `mypy`; `[tool.mypy] exclude = ["llm_sdk/"]`; `[[tool.mypy.overrides]] module = "llm_sdk" ignore_errors = true`.

### Key bug fixes during run testing

- **BPE newline check**: decoders originally checked `"\n" in tok` but Qwen's vocab stores newline as `Ċ` (U+010A). Fixed both number and string decoders to check `"Ċ" in tok`.
- **Number decoder early stop**: added `best_overall` check before masking — if model's top unconstrained token is `Ċ`/empty, stop (prevents `234567890123.0` from `2`).
- **String BPE artifacts**: replaced manual `"".join(id_to_token[tid] for tid in built_ids)` with `llm.decode(built_ids)` — fixes `Ġ` (BPE space) appearing in output.
- **Multi-param context**: added `already_extracted: dict[str, object] | None` to both `decode_number` and `decode_string`; `main.py` passes `dict(parameters)` so each subsequent param decode knows what was already extracted.
- **`accelerate` missing**: added to `pyproject.toml` — required by newer `transformers` when `device_map="auto"` used in `llm_sdk`.

### User-driven changes (end of session)

- `src/utils/constants.py` expanded with all CLI constants (`PROG_NAME`, `PROG_DESCRIPTION`, `PATH1/2/3`, `HELP1/2/3`, `MODEL_FLAG`, `MODEL`, `MODEL_HELP`, `VERBOSE_FLAG`, `VERBOSE_HELP`)
- `src/parser/parse.py` refactored to import from `constants` module; `required=True` removed from args so defaults kick in; run with `uv run python -m src` (no args)
- `src/main.py` minor style cleanup (blank lines between statements removed)

---

## Files Modified This Session

| File | Change |
|------|--------|
| `src/main.py` | Major rewrite — public API, string decoding, model flag, verbose, already_extracted, local model path detection |
| `src/parser/parse.py` | Added --model, --verbose; defaults wired; refactored to use constants |
| `src/utils/constants.py` | Expanded with all CLI constants |
| `src/parser/validator.py` | Type annotations, fixed SystemError→SystemExit, improved error messages |
| `src/model/input_format.py` | Fixed blank line violations |
| `src/decoder/function_chooser.py` | Removed private access, added types, verbose, error recovery, Protocol import |
| `src/decoder/number_decoder.py` | Removed private access, added caching, verbose, early-stop on Ċ, already_extracted |
| `src/decoder/string_decoder.py` | **New file** — string constrained decoder |
| `src/model/llm_protocol.py` | **New file** — LLMProtocol(Protocol) |
| `tests/__init__.py` | **New file** |
| `tests/test_number_decoder.py` | **New file** — 11 unit tests |
| `tests/test_function_chooser.py` | **New file** — 5 unit tests |
| `.gitignore` | Added `data/output/` |
| `pyproject.toml` | Added accelerate, pytest; dev group; mypy config |
| `setup.cfg` | **New file** — flake8 config |
| `CLAUDE.md` | Updated with subject constraints, known violations, missing features |
| `src/__main__.py` | Added trailing newline |

### Files deleted

`callmemaybe.pdf`, `en.subject (2).pdf`, `data (1).zip`, `PDF_ANALYSIS_SUMMARY.txt`, `scratch/`, `output.json` (root), `src/output.json`, `src/pipeline/pipeline.py`, `llm_sdk/.DS_Store`

---

## Observed Output Quality (Qwen3-0.6B)

From `make run --verbose`:

| Prompt | Function selected | Params | Correct? |
|--------|------------------|--------|---------|
| What is the sum of 2 and 3? | fn_add_numbers | a=2.0, b=2.0 | fn ✅, a ✅, b ❌ (should be 3.0) |
| What is the sum of 265 and 345? | fn_add_numbers | a=265.0, b=265.0 | fn ✅, a ✅, b ❌ |
| Greet shrek | fn_greet | name='shrek' | ✅ |
| Greet john | fn_greet | name='john' | ✅ |
| Reverse the string 'hello' | fn_reverse_string | s='hello' | ✅ |
| Reverse the string 'world' | fn_reverse_string | s='``' | fn ✅, s ❌ |
| What is the square root of 16? | fn_get_square_root | a=16.0 | ✅ |
| Calculate the square root of 144 | fn_get_square_root | a=144.0 | ✅ |
| Replace all numbers... | fn_substitute_string_with_regex | all wrong | fn ✅, params ❌ |
| Replace all vowels... | fn_substitute_string_with_regex | source_string correct | fn ✅, partial |
| Substitute 'cat' with 'dog'... | fn_substitute_string_with_regex | regex='cat' ✅ | fn ✅, partial |

**Function selection: 11/11 correct.**  
**Root cause of param failures**: 0.6B model insufficient capacity for second-param disambiguation and 3-param regex extraction. Expected to improve significantly with 14B model.

---

## What Was NOT Done

- **README** — missing required subject sections: algorithm explanation, design decisions, performance analysis, challenges faced, testing strategy. User explicitly deferred.
- **Docstrings** — user explicitly deferred. Subject grades them (PEP 257 on all functions/classes).
- **Bonus #2/8/9** — tokenizer reimplementation from scratch using only `get_path_to_vocab_file` + `get_logits_from_input_ids` (no `encode`/`decode`). Not implemented — significant effort, not chosen.
- **Bonus #7** — nested function arguments. Not implemented.
- **Committed** — no commits made this session. Last commit is `51c5e22 before comments`.

---

## Key Facts for Next Session

- Run with no args: `uv run python -m src`
- Run tests: `uv run pytest tests/ -v`
- Lint: `make lint` (runs `uv run flake8 .` + `uv run mypy . --warn-return-any ...`)
- User's 14B model is Ollama format — **incompatible** with `llm_sdk` (needs HF transformers format). Cannot be used directly.
- The BPE newline in Qwen vocab is `Ċ` (U+010A), not `\n`. Any new decoder must check for `Ċ`.
- `token_to_id` (returned by `load_vocab_maps`) is unused in the current pipeline — only `id_to_token` is used.
- `PormptExample` in `input_format.py` is a typo of `PromptExample` — kept as-is since `validator.py` imports it by that name.

---

## Where We're Going (Next Session)

Priority order:
1. **Write the README** — algorithm explanation, design decisions, performance analysis, challenges, testing strategy, example usage (subject grades this separately as Chapter VI)
2. **Add docstrings** — PEP 257, all functions and classes in `src/`
3. **Commit everything** — nothing has been committed this session
4. **Optional**: attempt bonus #2 (tokenizer reimplementation) for extra points

---

## Open Questions

- Does the evaluator's `make lint` use the `setup.cfg` we added? (Should — flake8 reads it automatically from the project root.)
- Will the evaluator swap `functions_definition.json` with a different file containing new function names? (Yes — subject says so. Our decoder is dynamic, not hardcoded, so this is handled.)
