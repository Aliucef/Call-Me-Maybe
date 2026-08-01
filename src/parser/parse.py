import argparse
from src.utils.constants import FLAG1, FLAG2, FLAG3
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    params = argparse.ArgumentParser(
        prog="function-calling",
        description="Generate function-calling JSON results from prompts."
    )

    params.add_argument(FLAG1, type=Path,
                        default=Path("data/input/functions_definition.json"),
                        help="Path to functions_definition.json")
    params.add_argument(FLAG2, type=Path,
                        default=Path("data/input/function_calling_tests.json"),
                        help="Path to function_calling_tests.json")
    params.add_argument(FLAG3, type=Path,
                        default=Path("data/output/function_calling_results.json"),
                        help="Path to write function_calling_results.json")
    params.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace model identifier (default: Qwen/Qwen3-0.6B)",
    )
    params.add_argument(
        "--verbose",
        action="store_true",
        help="Print token-level decoding steps for each prompt",
    )

    return params.parse_args()
