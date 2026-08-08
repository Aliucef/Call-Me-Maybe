# Re-exports every exception type at package level so other modules can do
# `from src.exceptions import FileNotFoundError, InvalidJSONError, ...`
# instead of reaching into each submodule individually.
from src.exceptions.input_file_error import InputFileError
from src.exceptions.file_not_found_error import FileNotFoundError
from src.exceptions.invalid_json_error import InvalidJSONError
from src.exceptions.not_an_array_error import NotAnArrayError
from src.exceptions.schema_validation_error import SchemaValidationError

# __all__ controls what `from src.exceptions import *` would expose, and
# documents the intended public surface of this package.
__all__ = [
    "InputFileError",
    "FileNotFoundError",
    "InvalidJSONError",
    "NotAnArrayError",
    "SchemaValidationError",
]
