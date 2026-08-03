from pathlib import Path

from src.exceptions.input_file_error import InputFileError


class SchemaValidationError(InputFileError):
    """Raised when JSON data does not match the expected Pydantic schema."""

    def __init__(self, path: Path, detail: str) -> None:
        """Set error message for schema mismatch."""
        super().__init__(f"invalid schema in {path}:\n{detail}")
