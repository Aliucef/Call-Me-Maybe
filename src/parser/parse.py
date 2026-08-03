import argparse
from pathlib import Path

import src.utils.constants as const


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    params = argparse.ArgumentParser(
        prog=const.PROG_NAME,
        description=const.PROG_DESCRIPTION,
    )
    params.add_argument(const.FLAG1, type=Path,
                        default=Path(const.PATH1),
                        help=const.HELP1)
    params.add_argument(const.FLAG2, type=Path,
                        default=Path(const.PATH2),
                        help=const.HELP2)
    params.add_argument(const.FLAG3, type=Path,
                        default=Path(const.PATH3),
                        help=const.HELP3)
    params.add_argument(
        const.MODEL_FLAG,
        default=const.MODEL,
        help=const.MODEL_HELP,
    )
    params.add_argument(
        const.VERBOSE_FLAG,
        action="store_true",
        default=False,
        help=const.VERBOSE_HELP,
    )
    return params.parse_args()
