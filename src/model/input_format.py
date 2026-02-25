from pydantic import BaseModel
from typing import Any

class ParameterSchema(BaseModel):
    type: str


class FunctionDef(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, ParameterSchema]
    returns: dict[str, Any] | None = None


class OutputDict(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]


class Output(BaseModel):
    obj: list[OutputDict]

class PormptExample(BaseModel):
    prompt: str
