import argparse
from src.utils.constants import FLAG1, FLAG2, FLAG3
from pathlib import Path

def parse_args():
    params = argparse.ArgumentParser(
        prog="function-calling",
        description="Generate function-calling JSON results from prompts."
    )

    params.add_argument(FLAG1, required=True, type=Path,
                   help="Path to function_definitions.json")

    params.add_argument(FLAG2, required=True, type=Path,
                   help="Path to function_calling_tests.json")

    params.add_argument(FLAG3, required=True, type=Path,
                   help="Path to write function_calling_results.json")

    return params.parse_args()
