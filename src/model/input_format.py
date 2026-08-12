from pydantic import BaseModel
from typing import Any


class ParameterSchema(BaseModel):
    """Type of a single function parameter."""

    type: str


class FunctionDef(BaseModel):
    """One entry from functions_definition.json."""

    name: str
    description: str | None = None
    parameters: dict[str, ParameterSchema]
    returns: dict[str, Any] | None = None


class OutputDict(BaseModel):
    """One resolved result written to function_calling_results.json."""

    prompt: str
    name: str
    parameters: dict[str, Any]


class PormptExample(BaseModel):
    """One entry from function_calling_tests.json."""

    prompt: str
