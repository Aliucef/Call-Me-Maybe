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

**Where:** `src/decoder/function_chooser.py:choose_function_name()` lines 65–73

When the candidate set is exhausted before a single name is resolved (ambiguous tokens), the decoder falls back to scoring every function name against the current logits and returns the highest-scoring one — instead of crashing or returning an arbitrary result.

---

## 3. Performance optimizations (caching)

**Subject:** *Performance optimizations (caching, batching)*

**Where:** `src/decoder/number_decoder.py` — module-level `_allowed_number_cache: dict[str, set[int]]`
`src/decoder/number_decoder.py:allowed_next_number_tokens()` — returns from cache on hit

`allowed_next_number_tokens` is a pure function of the partially-built number string and the vocabulary. Results are cached on first call and reused across all prompts in a run, eliminating repeated vocab scans.

---

## 4. Comprehensive test suite

**Subject:** *Comprehensive test suite*

**Where:** `tests/test_number_decoder.py` — 11 tests
`tests/test_function_chooser.py` — 5 tests

16 tests total. All use a `MockLLM` that satisfies `LLMProtocol` without downloading any model.

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
