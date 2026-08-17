# Bonus Features

All 5 bonuses are implemented and working. Listed against the subject's bonus checklist (Chapter VII).

---

## 1. Multiple LLM model support

**Subject:** *Support for multiple LLM models beyond Qwen/Qwen3-0.6B*

**Where:** `src/utils/constants.py` — `MODEL_FLAG`, `MODEL`, `MODEL_HELP`
`src/parser/parse.py` — `--model` argument
`src/main.py:setup_llm()` — `Small_LLM_Model(model_name=model)`

**How to use:**
```bash
uv run python -m src --model Qwen/Qwen3-1.7B
uv run python -m src --model /path/to/local/model
```

---

## 2. Advanced error recovery

**Subject:** *Advanced error recovery mechanisms*

**Where:** `src/decoder/candidate_chooser.py:choose_from_candidates()` — the `if not candidates:` block (currently lines 82–94)

This is the one shared decoding primitive every decoder (function name, number, string, boolean) funnels through. If narrowing ever eliminated every candidate, instead of crashing it falls back to scoring every original candidate by summing the logits of its own token ids and returns the highest-scoring one. In the current pipeline every caller already filters out an empty candidate set before calling in, so this specifically guards the shared primitive itself rather than any one caller — see `tests/test_candidate_chooser.py::test_empty_candidates_raises` for the one remaining edge case (an empty dict from the start) that still raises rather than recovers.

---

## 3. Performance optimizations (caching)

**Subject:** *Performance optimizations (caching, batching)*

**Where:** `src/decoder/number_decoder.py` — module-level `_number_encode_cache: dict[str, list[int]]`
`src/decoder/number_decoder.py:encode_number_candidate()` — returns from cache on hit

Encoding a numeric literal (e.g. `"2"`) to token ids is a pure function of the string and the (fixed, per-run) tokenizer vocabulary. Prompts across the same run frequently share literals, so results are cached on first call and reused for the rest of the run, skipping repeated tokenizer calls for values already seen. Covered by `tests/test_number_decoder.py::TestEncodeNumberCandidateCache`.

---

## 4. Comprehensive test suite

**Subject:** *Comprehensive test suite*

**Where:** `tests/test_function_chooser.py` — 5 tests
`tests/test_number_decoder.py` — 11 tests
`tests/test_string_decoder.py` — 14 tests
`tests/test_boolean_decoder.py` — 4 tests
`tests/test_candidate_chooser.py` — 5 tests

39 tests total, covering every decoder module plus the shared `choose_from_candidates()` primitive (including a documented edge case in prefix-collision behaviour and the empty-candidates guard). All use a shared `MockLLM` (`tests/conftest.py`) that satisfies `LLMProtocol` without downloading any model.

```bash
uv run pytest tests/ -v
```

---

## 5. Visualization of the generation process

**Subject:** *Visualization of the generation process*

**Where:** `src/utils/constants.py` — `VERBOSE_FLAG`, `VERBOSE_HELP`
`src/parser/parse.py` — `--verbose` argument
`src/decoder/function_chooser.py:choose_function_name()` — prints top-3 candidates at each step
`src/decoder/number_decoder.py:decode_number()` — prints built string at each step
`src/decoder/string_decoder.py:decode_string()` — prints token at each step
`src/main.py:process_prompt()` — prints prompt header when verbose

**How to use:**
```bash
uv run python -m src --verbose
```

---

## Not implemented

| Bonus | Reason |
|-------|--------|
| Recoding the tokenizer (avoid `encode`/`decode`) | High effort — requires reimplementing BPE from scratch using only `get_path_to_vocab_file` and `get_logits_from_input_ids` |
| Support for complex nested function arguments | Not in current input schema |
| Public tokenizer `encode`/`decode` reimplementation | Same as recoding bonus above |
| Demonstration of encoding/decoding integration | Tied to the reimplementation bonus |
