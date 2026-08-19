PROG_NAME = "function-calling"
PROG_DESCRIPTION = "Generate function-calling JSON results from prompts."

FUNCTIONS_FLAG = "--functions_definition"
PATH1 = "data/input/functions_definition.json"
HELP1 = "Path to functions_definition.json"

INPUT_FLAG = "--input"
PATH2 = "data/input/function_calling_tests.json"
HELP2 = "Path to function_calling_tests.json"

OUTPUT_FLAG = "--output"
PATH3 = "data/output/function_calling_results.json"
HELP3 = "Path to write function_calling_results.json"

MODEL_FLAG = "--model"
MODEL = "Qwen/Qwen3-0.6B"
MODEL_HELP = "HuggingFace model identifier (default: Qwen/Qwen3-0.6B)"

VERBOSE_FLAG = "--verbose"
VERBOSE_HELP = "Print token-level decoding steps for each prompt"
