# This file centralizes every CLI flag name, default path, and help string
# used by src/parser/parse.py. Keeping them here (instead of inline in
# parse.py) means argparse's setup code stays readable and there is a single
# place to change a default path or flag name.

# PROGRAM NAME AND DESCRIPTION
# These two show up when the user runs `python -m src --help`.
PROG_NAME = "function-calling"                     # shown as the program name in --help usage line
PROG_DESCRIPTION = "Generate function-calling JSON results from prompts."  # summary in --help

# FIRST ARGUMENT: where the function catalog lives
# Example: functions_definition.json contains fn_add_numbers, fn_greet, etc.
FLAG1 = "--functions_definition"                   # e.g. `--functions_definition path.json`
PATH1 = "data/input/functions_definition.json"     # default value if the flag is omitted
HELP1 = "Path to functions_definition.json"        # shown next to the flag in --help

# SECOND ARGUMENT: where the prompts to answer live
# Example: function_calling_tests.json contains {"prompt": "What is the sum of 2 and 3?"}
FLAG2 = "--input"
PATH2 = "data/input/function_calling_tests.json"
HELP2 = "Path to function_calling_tests.json"

# THIRD ARGUMENT: where the final answers get written
# Example: [{"prompt": "...", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}]
FLAG3 = "--output"
PATH3 = "data/output/function_calling_results.json"
HELP3 = "Path to write function_calling_results.json"

# MODEL ARGUMENT: which HuggingFace model llm_sdk.Small_LLM_Model should load
MODEL_FLAG = "--model"
MODEL = "Qwen/Qwen3-0.6B"                          # small model, fast enough to run on CPU
MODEL_HELP = "HuggingFace model identifier (default: Qwen/Qwen3-0.6B)"

# VERBOSE ARGUMENT: a boolean switch (no value expected after it)
# When set, decoders print each token they commit to, so during the defense
# you can run with --verbose to show the constrained decoding loop live.
VERBOSE_FLAG = "--verbose"
VERBOSE_HELP = "Print token-level decoding steps for each prompt"
