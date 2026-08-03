from pathlib import Path

from src.exceptions.input_file_error import InputFileError


class InvalidJSONError(InputFileError):
    """Raised when a file contains malformed JSON."""

    def __init__(self, path: Path, detail: str) -> None:
        """Set error message for JSON parse failure."""
        super().__init__(f"invalid JSON in {path}: {detail}")
