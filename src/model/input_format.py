from pydantic import BaseModel
from typing import Any


# pydantic's BaseModel gives us automatic validation: if the JSON we load
# does not match these field names/types, pydantic raises a ValidationError
# instead of the program silently working with wrong data. This is how the
# project enforces "no crashing on bad input" — see src/parser/validator.py,
# which turns that ValidationError into a friendly SchemaValidationError.


class ParameterSchema(BaseModel):
    """Describes one parameter of a function, e.g. {"type": "number"}."""

    # Only "number" and "string" are meaningful to the decoders, but the
    # schema itself just stores whatever string is in the JSON — the
    # decoder dispatch (see src/main.py process_prompt) is what interprets it.
    type: str


class FunctionDef(BaseModel):
    """One entry of functions_definition.json.

    Example JSON:
        {
          "name": "fn_add_numbers",
          "description": "Add two numbers together and return their sum.",
          "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
          "returns": {"type": "number"}
        }
    """

    name: str                        # e.g. "fn_add_numbers" — what the LLM must output
    description: str | None = None   # optional human-readable text shown to the LLM as context
    # dict keyed by parameter name -> its schema, e.g. {"a": ParameterSchema(type="number")}
    parameters: dict[str, ParameterSchema]
    returns: dict[str, Any] | None = None  # unused by pipeline logic, kept for reference


class OutputDict(BaseModel):
    """One entry written to function_calling_results.json.

    Example JSON produced for prompt "What is the sum of 2 and 3?":
        {"prompt": "What is the sum of 2 and 3?",
         "name": "fn_add_numbers",
         "parameters": {"a": 2.0, "b": 3.0}}
    """

    prompt: str                  # the original prompt, echoed back for traceability
    name: str                    # the function name chosen by choose_function_name()
    parameters: dict[str, Any]   # param name -> decoded value (float for numbers, str for strings)


class Output(BaseModel):
    """Wrapper model for a full list of OutputDict — currently unused by
    main.py (which serializes the list directly), kept for convenience if a
    single top-level object is ever needed instead of a bare JSON array."""

    obj: list[OutputDict]


class PormptExample(BaseModel):
    """One entry of function_calling_tests.json, e.g. {"prompt": "What is the sum of 2 and 3?"}.

    Note: the name has a typo ("Prompt" -> "Pormpt") but it's kept as-is
    since it's referenced by that exact name elsewhere in the codebase
    (src/main.py, src/parser/validator.py).
    """

    prompt: str
