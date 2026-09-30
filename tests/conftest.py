"""Fixtures shared by the test modules.

Everything under test here is a pure function, so no fixture builds a
model definition or reads a weight file, and the whole suite finishes
in about a second.
"""

from collections.abc import Callable

import pytest

from call_me_maybe import generator
from call_me_maybe.models.function_definition import (
    FunctionDefinition,
    JsonType,
)
from call_me_maybe.models.function_lists import FunctionDefinitionList

#: Builds a definition list holding a single one-parameter function.
OneParameter = Callable[[str, str], FunctionDefinitionList]


def function(name: str, parameters: dict[str, JsonType]) -> FunctionDefinition:
    """One function definition, with a description to fill the field."""
    return FunctionDefinition(
        name=name,
        description=f"The {name} function.",
        parameters=parameters,
        returns={"type": "string"},
    )


def functions(*defs: FunctionDefinition) -> FunctionDefinitionList:
    """Wrap definitions in the list the grammar is handed."""
    return FunctionDefinitionList(list(defs))


@pytest.fixture
def definitions() -> FunctionDefinitionList:
    """One function per declared type, plus one the grammar has not seen.

    `fn_now` declares no parameters at all, so the empty-object case
    is covered by the same list as every other type.
    """
    return functions(
        function("fn_greet", {"name": {"type": "string"}}),
        function(
            "fn_add",
            {"a": {"type": "number"}, "b": {"type": "number"}},
        ),
        function(
            "fn_sub",
            {
                "source_string": {"type": "string"},
                "regex": {"type": "string"},
                "replacement": {"type": "string"},
            },
        ),
        function("fn_flag", {"on": {"type": "boolean"}}),
        function("fn_count", {"n": {"type": "integer"}}),
        function("fn_ratio", {"x": {"type": "float"}}),
        function("fn_items", {"items": {"type": "array"}}),
        function("fn_config", {"cfg": {"type": "object"}}),
        function("fn_unknown", {"x": {"type": "quantum"}}),
        function("fn_now", {}),
    )


@pytest.fixture
def one_parameter() -> OneParameter:
    """Build a definition list with one function and one parameter.

    Used to exercise a single declared type, so the type rules are
    tested against one `_VALUE_TYPES` entry at a time.
    """

    def build(name: str, declared: str) -> FunctionDefinitionList:
        return functions(function(name, {"value": {"type": declared}}))

    return build


@pytest.fixture(autouse=True)
def no_model_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn any attempt to build the model into a loud failure.

    What is under test is the tokenizer and the grammar, not the
    weights, so a test that reaches for `Small_LLM_Model` is a test
    that would make the suite slow and network-dependent. Failing here
    says so instead of quietly downloading a gigabyte.
    """

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("this suite must not load the model")

    monkeypatch.setattr(generator, "Small_LLM_Model", refuse)
