
from pathlib import Path
from pydantic import TypeAdapter, ValidationError
from src.model.input_format import FunctionDef, PormptExample
import json


def load_function_definitions(path):
    try:
        with open(path, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        raise SystemExit(f"[ERROR] file not found: {path}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"[ERROR] invalid JSON in {path}: {e}")

    if not isinstance(data, list):
        raise SystemExit(f"[ERROR] {path} must contain a JSON array")

    try:
        return TypeAdapter(list[FunctionDef]).validate_python(data)
    except ValidationError as e:
        raise SystemExit(
            f"[ERROR] invalid function definition schema in {path}:\n{e}"
        )

def load_prompt_examples(path : Path):
    try:
        with open(path, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        raise SystemExit("File not found error")
    except json.JSONDecodeError:
        raise SystemError("json error")

    if not isinstance(data , list):
        raise SystemExit("must contain json array")
    try:
        return TypeAdapter(list[PormptExample]).validate_python(data)
    except ValidationError as e:
        raise SystemError(f"fail structure {e}")
