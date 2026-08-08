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
        # "r" mode = text mode; json.load reads the whole file and parses it.
        with open(path, "r") as file:
            data = json.load(file)
    except OSError:
        # Covers file-not-found, permission errors, etc. Converted to our
        # own exception type so main.py only needs to catch InputFileError.
        raise FileNotFoundError(path)
    except json.JSONDecodeError as e:
        # Malformed JSON syntax (trailing comma, unclosed bracket, ...).
        raise InvalidJSONError(path, str(e))
    if not isinstance(data, list):
        # Valid JSON, but not shaped like `[...]` at the top level —
        # e.g. functions_definition.json accidentally written as a single
        # object instead of a list of objects.
        raise NotAnArrayError(path)
    # At this point `data` is a plain Python list of dicts/values with no
    # guarantee yet about what's inside each element — that's checked next
    # by the pydantic TypeAdapter in the two functions below.
    return data


def load_function_definitions(path: Path) -> list[FunctionDef]:
    """Load and validate a JSON array of function definitions."""
    # Example: for data/input/functions_definition.json this returns
    # [FunctionDef(name="fn_add_numbers", description="...",
    #              parameters={"a": ParameterSchema(type="number"),
    #                          "b": ParameterSchema(type="number")}, returns=...),
    #  FunctionDef(name="fn_greet", ...), ...]
    data = _load_json_array(path)
    try:
        # TypeAdapter(list[FunctionDef]) builds a pydantic validator for
        # "a list of FunctionDef objects" and .validate_python(data) checks
        # every element against that schema in one call, converting raw
        # dicts into typed FunctionDef instances (and their nested
        # ParameterSchema instances) along the way.
        return TypeAdapter(list[FunctionDef]).validate_python(data)
    except ValidationError as e:
        # Any element that doesn't match FunctionDef's fields/types
        # (missing "name", wrong type for "parameters", etc.) lands here.
        raise SchemaValidationError(path, str(e))


def load_prompt_examples(path: Path) -> list[PormptExample]:
    """Load and validate a JSON array of prompt examples."""
    # Example: for data/input/function_calling_tests.json this returns
    # [PormptExample(prompt="What is the sum of 2 and 3?"),
    #  PormptExample(prompt="Greet shrek"), ...]
    data = _load_json_array(path)
    try:
        return TypeAdapter(list[PormptExample]).validate_python(data)
    except ValidationError as e:
        raise SchemaValidationError(path, str(e))
