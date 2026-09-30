import json
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar

from call_me_maybe.json_parser import JSONToken, JSONTokenizer, JSONTokenType
from call_me_maybe.models.function_definition import FunctionDefinition
from call_me_maybe.models.function_lists import FunctionDefinitionList


class JsonState(Enum):
    """The steps of a call, in the order they must come."""

    Start = 0
    Name = 1
    NameColon = 2
    NameValue = 3
    NameComma = 4
    Parameters = 5
    ParametersColon = 6
    ParametersOpen = 7
    ParameterKey = 8
    ParameterColon = 9
    ParameterValue = 10
    ParameterComma = 11
    Close = 12
    Done = 13


@dataclass
class _Call:
    """What the walk has worked out about the call so far.

    `remaining` holds the parameters of `function` that still have to
    show up; the key check empties it in place, one key at a time.
    """

    function: FunctionDefinition | None = None
    remaining: set[str] = field(default_factory=set)
    parameter: str | None = None


class JSONGrammar:
    """Tells whether a text can still become a valid function call.

    Nothing but the call itself is accepted: the text must be the
    format the system prompt asks for, from the opening brace on.

    {
        "name": "<function name>",
        "parameters": {
            "<parameter>": <value>
        }
    }

    The name must be one of `functions`, and every parameter of that
    function must be filled with a value of the type it declares.
    """

    # The steps whose only rule is "this exact token type, then move
    # on". The steps that have to decide something are in the walk.
    _NEXT: ClassVar[dict[JsonState, tuple[JSONTokenType, JsonState]]] = {
        JsonState.Start: (JSONTokenType.BraceOpen, JsonState.Name),
        JsonState.NameColon: (JSONTokenType.Colon, JsonState.NameValue),
        JsonState.NameComma: (JSONTokenType.Comma, JsonState.Parameters),
        JsonState.ParametersColon: (
            JSONTokenType.Colon,
            JsonState.ParametersOpen,
        ),
        JsonState.ParametersOpen: (
            JSONTokenType.BraceOpen,
            JsonState.ParameterKey,
        ),
        JsonState.ParameterColon: (
            JSONTokenType.Colon,
            JsonState.ParameterValue,
        ),
        JsonState.Close: (JSONTokenType.BraceClose, JsonState.Done),
    }

    # The token types each declared type accepts. Null is always
    # allowed: the system prompt asks for it when a required value is
    # missing. Any other declared type takes any JSON value.
    _VALUE_TYPES: ClassVar[dict[str, frozenset[JSONTokenType]]] = {
        "string": frozenset({JSONTokenType.String, JSONTokenType.Null}),
        "number": frozenset({JSONTokenType.Number, JSONTokenType.Null}),
    }
    _ANY_VALUE: ClassVar[frozenset[JSONTokenType]] = frozenset(JSONTokenType)

    @staticmethod
    def valid_prefix(text: str, functions: FunctionDefinitionList) -> bool:
        """True if `text` can still become a valid function call.

        An unfinished trailing word is accepted, as long as it could
        grow into a valid token.
        """
        return JSONGrammar.__state(text, functions) is not None

    @staticmethod
    def is_closed(text: str, functions: FunctionDefinitionList) -> bool:
        """True if `text` already is a complete function call."""
        return JSONGrammar.__state(text, functions) is JsonState.Done

    @staticmethod
    def __state(
        text: str, functions: FunctionDefinitionList
    ) -> JsonState | None:
        """The step `text` has reached, or None if it is off the call."""
        try:
            tokens = JSONTokenizer.tokenize(text)
        except ValueError:
            return None
        return JSONGrammar.__walk(tokens, functions, text)

    @staticmethod
    def __walk(
        tokens: list[JSONToken],
        functions: FunctionDefinitionList,
        text: str,
    ) -> JsonState | None:
        """Walk `tokens` through the steps, from `Start` onwards.

        Returns the step reached, or None as soon as a token does not
        fit the call. An unfinished trailing word is accepted if it
        could still grow into a valid token.
        """
        state = JsonState.Start
        call = _Call()
        last = len(tokens) - 1
        for index, token in enumerate(tokens):
            if token.type is JSONTokenType.Incomplete:
                if index != last or not JSONGrammar.__fragment_fits(
                    state, token, text, functions, call
                ):
                    return None
                continue
            next_state = JSONGrammar.__step(state, token, functions, call)
            if next_state is None:
                return None
            state = next_state
        return state

    @staticmethod
    def __step(
        state: JsonState,
        token: JSONToken,
        functions: FunctionDefinitionList,
        call: _Call,
    ) -> JsonState | None:
        """The step `token` leads to, or None if it does not fit.

        The steps whose only rule is "this exact token type, then move
        on" are read off `_NEXT`. The ones that have to decide something
        are below.
        """
        rule = JSONGrammar._NEXT.get(state)
        if rule is not None:
            expected, following = rule
            if token.type is not expected:
                return None
            return following

        match state:
            case JsonState.Name:
                if not JSONGrammar.__is_key(token, "name"):
                    return None
                return JsonState.NameColon
            case JsonState.NameValue:
                # The name picks the function, and with it the
                # parameters that must all show up below.
                if token.type is not JSONTokenType.String:
                    return None
                function = next(
                    (f for f in functions.root if f.name == token.value),
                    None,
                )
                if function is None:
                    return None
                call.function = function
                call.remaining = set(function.parameters)
                return JsonState.NameComma
            case JsonState.Parameters:
                if not JSONGrammar.__is_key(token, "parameters"):
                    return None
                return JsonState.ParametersColon
            case JsonState.ParameterKey:
                after = JSONGrammar.__after_key(token, call.remaining)
                if after is None:
                    return None
                if after is JsonState.ParameterColon:
                    call.parameter = token.value
                return after
            case JsonState.ParameterValue:
                if not JSONGrammar.__value_allowed(call, token):
                    return None
                return JsonState.ParameterComma
            case JsonState.ParameterComma:
                if call.remaining and token.type is JSONTokenType.Comma:
                    return JsonState.ParameterKey
                if (
                    not call.remaining
                    and token.type is JSONTokenType.BraceClose
                ):
                    return JsonState.Close
                return None
            case JsonState.Done:
                # The call is already closed: nothing can follow.
                return None
        return None

    @staticmethod
    def __after_key(token: JSONToken, remaining: set[str]) -> JsonState | None:
        """The step after a parameter key, or None if it does not fit.

        A key is accepted once, and only if the function declares it.
        With every parameter in, the object can close instead.
        """
        if token.type is JSONTokenType.String:
            if token.value not in remaining:
                return None
            remaining.remove(token.value)
            return JsonState.ParameterColon
        if not remaining and token.type is JSONTokenType.BraceClose:
            return JsonState.Close
        return None

    @staticmethod
    def __is_key(token: JSONToken, key: str) -> bool:
        """True if `token` is the exact string key `key`."""
        return token.type is JSONTokenType.String and token.value == key

    @staticmethod
    def __value_allowed(call: _Call, token: JSONToken) -> bool:
        """True if `token` is a value of the declared parameter type."""
        if call.function is None or call.parameter is None:
            return False
        declared = call.function.parameters[call.parameter]["type"]
        return token.type in JSONGrammar._VALUE_TYPES.get(
            declared, JSONGrammar._ANY_VALUE
        )

    @staticmethod
    def __fragment_fits(
        state: JsonState,
        token: JSONToken,
        text: str,
        functions: FunctionDefinitionList,
        call: _Call,
    ) -> bool:
        """True if the unfinished `token` could still grow into a valid
        token in `state`.

        A fragment counts only while the string it belongs to is really
        open in `text`: without its opening quote it could never be
        completed.
        """
        quoted = text.endswith('"' + token.value)
        if state is JsonState.ParameterValue:
            return JSONGrammar.__value_grows(call, token.value, quoted)
        expected = JSONGrammar.__expected_text(
            state, functions, call.remaining
        )
        return quoted and any(
            word.startswith(token.value) for word in expected
        )

    @staticmethod
    def __expected_text(
        state: JsonState,
        functions: FunctionDefinitionList,
        remaining: set[str],
    ) -> tuple[str, ...]:
        """The strings a fragment in `state` could still grow into."""
        if state is JsonState.Name:
            return ("name",)
        if state is JsonState.Parameters:
            return ("parameters",)
        if state is JsonState.NameValue:
            return tuple(f.name for f in functions.root)
        if state is JsonState.ParameterKey:
            return tuple(remaining)
        return ()

    @staticmethod
    def __value_grows(call: _Call, value: str, quoted: bool) -> bool:
        """True if `value` could still grow into a value of the type
        the parameter declares."""
        if call.function is None or call.parameter is None:
            return False
        match call.function.parameters[call.parameter]["type"]:
            case "string":
                # Any text can still be a string value.
                return quoted
            case "number":
                return not quoted and JSONGrammar.__could_be_number(value)
        return False

    @staticmethod
    def __could_be_number(value: str) -> bool:
        """True if `value` could still grow into a JSON number."""
        try:
            parsed = json.loads(value + "0")
        except json.JSONDecodeError:
            return False
        # `type()` instead of `isinstance()`: bool is a subclass of int.
        return type(parsed) in (int, float)
