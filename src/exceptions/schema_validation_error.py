from pathlib import Path

from src.exceptions.input_file_error import InputFileError


class SchemaValidationError(InputFileError):
    """Raised when JSON data does not match the expected Pydantic schema.

    Triggered when the JSON parses fine and is an array, but an element
    doesn't fit FunctionDef / PormptExample — e.g. a function entry missing
    "name", or "parameters.a.type" being a number instead of a string.
    """

    def __init__(self, path: Path, detail: str) -> None:
        """Set error message for schema mismatch."""
        # `detail` is str(the original pydantic ValidationError), which
        # already lists every field that failed and why.
        super().__init__(f"invalid schema in {path}:\n{detail}")
