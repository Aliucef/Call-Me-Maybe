import argparse
from pathlib import Path

import src.utils.constants as const


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    # ArgumentParser is the stdlib CLI builder. Giving it prog/description
    # only affects the text shown by `--help`; it has no effect on parsing.
    params = argparse.ArgumentParser(
        prog=const.PROG_NAME,
        description=const.PROG_DESCRIPTION,
    )
    # --functions_definition PATH (defaults to data/input/functions_definition.json)
    # type=Path means argparse converts the raw string straight into a
    # pathlib.Path object, so callers never deal with plain strings.
    params.add_argument(const.FLAG1, type=Path,
                        default=Path(const.PATH1),
                        help=const.HELP1)
    # --input PATH (defaults to data/input/function_calling_tests.json)
    # Example value after parsing: PosixPath('data/input/function_calling_tests.json')
    params.add_argument(const.FLAG2, type=Path,
                        default=Path(const.PATH2),
                        help=const.HELP2)
    # --output PATH (defaults to data/output/function_calling_results.json)
    params.add_argument(const.FLAG3, type=Path,
                        default=Path(const.PATH3),
                        help=const.HELP3)
    # --model NAME (defaults to "Qwen/Qwen3-0.6B")
    # No type= given, so this stays a plain str — it's passed straight into
    # Small_LLM_Model(model_name=...) in main.py.
    params.add_argument(
        const.MODEL_FLAG,
        default=const.MODEL,
        help=const.MODEL_HELP,
    )
    # --verbose is a flag with no argument: action="store_true" makes it
    # True when present on the command line, False (the default) otherwise.
    params.add_argument(
        const.VERBOSE_FLAG,
        action="store_true",
        default=False,
        help=const.VERBOSE_HELP,
    )
    # Reads sys.argv, applies the rules above, and returns a Namespace whose
    # attributes are named after the flags with leading dashes stripped and
    # remaining dashes turned into underscores, e.g. args.functions_definition,
    # args.input, args.output, args.model, args.verbose.
    return params.parse_args()
