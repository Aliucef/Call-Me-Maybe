from pathlib import Path

from src.exceptions.input_file_error import InputFileError


class InvalidJSONError(InputFileError):
    """Raised when a file contains malformed JSON.

    Triggered in src/parser/validator.py when json.load() raises
    json.JSONDecodeError, e.g. if functions_definition.json has a trailing
    comma or an unclosed brace.
    """

    def __init__(self, path: Path, detail: str) -> None:
        """Set error message for JSON parse failure."""
        # `detail` is str(the original JSONDecodeError), e.g.
        # "Expecting ',' delimiter: line 12 column 3 (char 210)"
        super().__init__(f"invalid JSON in {path}: {detail}")
