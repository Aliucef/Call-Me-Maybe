from pathlib import Path

from src.exceptions.input_file_error import InputFileError


class NotAnArrayError(InputFileError):
    """Raised when a JSON file does not contain a top-level array.

    Both input files must be JSON arrays (a list of function objects, or a
    list of prompt objects). If the top-level JSON value parses fine but is
    e.g. a single object `{...}` instead of `[{...}]`, this fires.
    """

    def __init__(self, path: Path) -> None:
        """Set error message for non-array JSON root."""
        super().__init__(f"{path} must contain a JSON array")
