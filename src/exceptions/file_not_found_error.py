from pathlib import Path

from src.exceptions.input_file_error import InputFileError


class FileNotFoundError(InputFileError):
    """Raised when an input file does not exist.

    Note: this shadows the Python builtin `FileNotFoundError` within this
    module's namespace — deliberate, since it's imported and used by name
    via `from src.exceptions import FileNotFoundError` elsewhere, and it
    IS conceptually a FileNotFoundError, just one specific to this project's
    input-loading flow (see src/parser/validator.py `_load_json_array`,
    which catches the builtin OSError and raises this instead).
    """

    def __init__(self, path: Path) -> None:
        """Set error message for missing file."""
        # Example: FileNotFoundError(Path("data/input/functions_definition.json"))
        # -> str(e) == "file not found: data/input/functions_definition.json"
        super().__init__(f"file not found: {path}")
