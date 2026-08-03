from pathlib import Path

from src.exceptions.input_file_error import InputFileError


class FileNotFoundError(InputFileError):
    """Raised when an input file does not exist."""

    def __init__(self, path: Path) -> None:
        """Set error message for missing file."""
        super().__init__(f"file not found: {path}")
