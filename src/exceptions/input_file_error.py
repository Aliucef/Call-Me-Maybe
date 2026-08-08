class InputFileError(Exception):
    """Base class for all input file errors.

    src/parser/validator.py never lets a raw OSError/JSONDecodeError/
    ValidationError escape — it always wraps them into one of this
    exception's subclasses (FileNotFoundError, InvalidJSONError,
    NotAnArrayError, SchemaValidationError) so src/main.py can catch a
    single type and print a clean "[ERROR] ..." message instead of a
    Python traceback. This is the mechanism behind the subject requirement
    "no unhandled exceptions."
    """
