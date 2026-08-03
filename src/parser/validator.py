import json
from pathlib import Path
from pydantic import TypeAdapter, ValidationError

from src.exceptions import (
    FileNotFoundError,
    InvalidJSONError,
    NotAnArrayError,
    SchemaValidationError,
)
from src.model.input_format import FunctionDef, PormptExample


def _load_json_array(path: Path) -> list[object]:
    """Read a file and return its contents parsed as a JSON array."""
    try:
        with open(path, "r") as file:
            data = json.load(file)
    except OSError:
        raise FileNotFoundError(path)
    except json.JSONDecodeError as e:
        raise InvalidJSONError(path, str(e))
    if not isinstance(data, list):
        raise NotAnArrayError(path)
    return data


def load_function_definitions(path: Path) -> list[FunctionDef]:
    """Load and validate a JSON array of function definitions."""
    data = _load_json_array(path)
    try:
        return TypeAdapter(list[FunctionDef]).validate_python(data)
    except ValidationError as e:
        raise SchemaValidationError(path, str(e))


def load_prompt_examples(path: Path) -> list[PormptExample]:
    """Load and validate a JSON array of prompt examples."""
    data = _load_json_array(path)
    try:
        return TypeAdapter(list[PormptExample]).validate_python(data)
    except ValidationError as e:
        raise SchemaValidationError(path, str(e))
