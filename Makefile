PYTHON := uv run python
MAIN := src/main.py
FUNCTIONS := data/input/functions_definition.json
INPUT := data/input/function_calling_tests.json
OUTPUT := data/output/function_calling_results.json

CACHE_DIR := $(CURDIR)/.cache
export UV_CACHE_DIR ?= $(CACHE_DIR)/uv
export TMPDIR ?= $(CACHE_DIR)/tmp
export HF_HOME ?= $(CACHE_DIR)/huggingface
export HF_HUB_DISABLE_XET ?= 1

.PHONY: install run debug clean lint lint-strict

install:
	mkdir -p $(UV_CACHE_DIR) $(TMPDIR) $(HF_HOME)
	uv sync

run:
	uv run python -m src --functions_definition $(FUNCTIONS) --input $(INPUT) --output $(OUTPUT)

debug:
	uv run python -m pdb -m src --functions_definition $(FUNCTIONS) --input $(INPUT) --output $(OUTPUT)

clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type d -name '.mypy_cache' -prune -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -prune -exec rm -rf {} +

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict
