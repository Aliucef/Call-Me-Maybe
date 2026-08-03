# PROGRAM NAME AND DESCRIPTION
PROG_NAME = "function-calling"
PROG_DESCRIPTION = "Generate function-calling JSON results from prompts."

# FIRST ARGUMENT
FLAG1 = "--functions_definition"
PATH1 = "data/input/functions_definition.json"
HELP1 = "Path to functions_definition.json"

# SECOND ARGUMENT
FLAG2 = "--input"
PATH2 = "data/input/function_calling_tests.json"
HELP2 = "Path to function_calling_tests.json"

# THIRD ARGUMENT
FLAG3 = "--output"
PATH3 = "data/output/function_calling_results.json"
HELP3 = "Path to write function_calling_results.json"

# MODEL ARGUMENT
MODEL_FLAG = "--model"
MODEL = "Qwen/Qwen3-0.6B"
MODEL_HELP = "HuggingFace model identifier (default: Qwen/Qwen3-0.6B)"

# VERBOSE ARGUMENT
VERBOSE_FLAG = "--verbose"
VERBOSE_HELP = "Print token-level decoding steps for each prompt"
