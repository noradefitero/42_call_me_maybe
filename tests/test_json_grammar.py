"""Unit tests for the grammar: `valid_prefix` and `is_closed`.

The grammar is the correctness claim of the project, and it is a pure
function of the text and the definition list, so it is driven here
directly rather than through the decoder.

A prefix is *valid* when the text could still grow into a complete,
schema-compliant call; it is *closed* when it already is one. Both
questions are asked of a text that is usually not JSON at all.
"""

import json

import pytest
from conftest import OneParameter

from call_me_maybe.json_grammar import JSONGrammar
from call_me_maybe.models.function_lists import FunctionDefinitionList

#: Answers the grammar has to accept, for the `definitions` fixture.
VALID_CALLS = [
    '{"name":"fn_greet","parameters":{"name":"shrek"}}',
    '{"name":"fn_add","parameters":{"a":2,"b":3}}',
    (
        '{"name":"fn_sub","parameters":{"source_string":"a","regex":"b",'
        '"replacement":"c"}}'
    ),
    (
        '{"name":"fn_sub","parameters":{"replacement":"c",'
        '"source_string":"a","regex":"b"}}'
    ),
    '{"name":"fn_flag","parameters":{"on":true}}',
    '{"name":"fn_count","parameters":{"n":-7}}',
    '{"name":"fn_now","parameters":{}}',
    # `null` is the documented answer for a required value the model
    # could not fill in, so every declared type has to accept it.
    '{"name":"fn_greet","parameters":{"name":null}}',
    '{"name":"fn_add","parameters":{"a":null,"b":null}}',
    '{"name":"fn_items","parameters":{"items":null}}',
]

#: Text the grammar has to turn down. The empty text is not here: with
#: nothing written yet, any answer is still possible.
INVALID_CALLS = [
    # nothing that opens an object
    "hello",
    "[1,2]",
    # a wrong or missing opening brace
    '"name":"fn_greet","parameters":{"name":"x"}}',
    "{]",
    # the first key is not `name`
    '{"Name":"fn_greet","parameters":{"name":"x"}}',
    '{"function":"fn_greet","parameters":{"name":"x"}}',
    '{"name " "fn_greet","parameters":{"name":"x"}}',
    # the function is not one of the definitions
    '{"name":"fn_nope","parameters":{}}',
    '{"name":"","parameters":{}}',
    '{"name":5,"parameters":{}}',
    # a missing colon or comma
    '{"name" "fn_greet","parameters":{"name":"x"}}',
    '{"name":"fn_greet" "parameters":{"name":"x"}}',
    '{"name":"fn_greet","parameters" {"name":"x"}}',
    # the second key is not `parameters`
    '{"name":"fn_greet","parmeters":{"name":"x"}}',
    '{"name":"fn_greet","args":{"name":"x"}}',
    # no `parameters` key at all
    '{"name":"fn_greet"}',
    '{"name":"fn_greet",}',
    # an extra key beside the two the format allows
    '{"name":"fn_greet","parameters":{"name":"x"},"zz":1}',
    '{"name":"fn_greet","parameters":{"name":"x"},"note":"hi"}',
    # a parameter left out, so the object closes too early
    '{"name":"fn_greet","parameters":{}}',
    '{"name":"fn_greet","parameters":{"name":"x",}}',
    # text after the closing brace
    '{"name":"fn_greet","parameters":{"name":"x"}} and then some',
    '{"name":"fn_greet","parameters":{"name":"x"}}{}',
]

#: Well-formed JSON that is still not a function call. A parsable
#: object is a necessary condition for an answer, not a sufficient one.
VALID_JSON_NOT_A_CALL = [
    '{"name":"fn_greet","parameters":{"name":"x"},"zz":1}',
    '{"name":"fn_greet"}',
    '{"name":"fn_nope","parameters":{}}',
    '{"name":"fn_greet","parameters":{"name":"x","extra":1}}',
    '{"name":"fn_greet","parameters":{"name":5}}',
    '{"name":"fn_greet","parmeters":{"name":"x"}}',
    '{"name":"fn_greet","parameters":{"name":"x"},"notes":null}',
    '{"result":5}',
]

#: Text that is neither valid JSON nor still growing: it looks like a
#: finished answer but cannot become one. None of it may be reported as
#: closed, and none of it may be called a valid prefix either.
MALFORMED_CASES = [
    (
        "an escape json does not define",
        '{"name":"fn_greet","parameters":{"name":"a\\db"}}',
    ),
    (
        "another escape json does not define",
        '{"name":"fn_greet","parameters":{"name":"a\\qb"}}',
    ),
    (
        "a hex escape json does not define",
        '{"name":"fn_greet","parameters":{"name":"a\\x41b"}}',
    ),
    (
        "a unicode escape cut short by the closing quote",
        '{"name":"fn_greet","parameters":{"name":"a\\u04"}}',
    ),
    (
        "a unicode escape whose last digit is the closing brace",
        '{"name":"fn_greet","parameters":{"name":"a\\u12b"}}',
    ),
    (
        "a unicode escape whose digits are not hex",
        '{"name":"fn_greet","parameters":{"name":"a\\uZZZZb"}}',
    ),
    (
        "a stray quote opening a second string",
        '{"name":"fn_greet","parameters":{"name":"a" "b"}}',
    ),
]

#: Text cut off part way through an answer. Not JSON, and not meant to
#: be: this is what the answer looks like at the step before the last,
#: so it has to stay a valid prefix.
UNTERMINATED_CASES = [
    (
        "a string with no closing quote",
        '{"name":"fn_greet","parameters":{"name":"unterminated}',
    ),
    (
        "an escaped quote at the end, so nothing closes the string",
        '{"name":"fn_greet","parameters":{"name":"a\\"}',
    ),
    (
        "a function name half typed",
        '{"name":"fn_gr',
    ),
]

MALFORMED = [text for _, text in MALFORMED_CASES]
UNTERMINATED = [text for _, text in UNTERMINATED_CASES]
NOT_JSON = MALFORMED + UNTERMINATED


# --------------------------------------------------------------------------
# Valid prefixes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("text", VALID_CALLS)
def test_a_complete_call_is_closed(
    text: str, definitions: FunctionDefinitionList
) -> None:
    assert JSONGrammar.is_closed(text, definitions)
    assert JSONGrammar.valid_prefix(text, definitions)


@pytest.mark.parametrize("text", VALID_CALLS)
def test_every_prefix_of_a_complete_call_is_valid(
    text: str, definitions: FunctionDefinitionList
) -> None:
    # The property constrained decoding rests on: the answer stays a
    # legal prefix the whole way it is written, so the decoder is never
    # asked to reject a token the finished answer depends on.
    for length in range(len(text) + 1):
        prefix = text[:length]
        assert JSONGrammar.valid_prefix(prefix, definitions), prefix


@pytest.mark.parametrize("text", VALID_CALLS)
def test_a_complete_call_is_closed_only_at_its_end(
    text: str, definitions: FunctionDefinitionList
) -> None:
    # Nothing short of the final `}` may count as closed, or generation
    # would stop in the middle of a call.
    for length in range(len(text)):
        assert not JSONGrammar.is_closed(text[:length], definitions), length


@pytest.mark.parametrize(
    "text",
    [
        # Whitespace is dropped by the tokenizer, so it is allowed
        # wherever it does not change what the grammar sees.
        '{ "name" : "fn_greet" , "parameters" : { "name" : "x" } }',
        '{\n\t"name": "fn_greet",\n\t"parameters": {"name": "x"}\n}',
        '{ "name":"fn_now" , "parameters" : { } }',
    ],
)
def test_whitespace_wherever_it_is_allowed(
    text: str, definitions: FunctionDefinitionList
) -> None:
    assert JSONGrammar.is_closed(text, definitions)


@pytest.mark.parametrize(
    "text",
    [
        '{"na',
        '{"nam',
        '{"name"',
        '{"name":',
        '{"name":"fn_gr',
        '{"name":"fn_greet"',
        '{"name":"fn_greet",',
        '{"name":"fn_greet","p',
        '{"name":"fn_greet","paramet',
        '{"name":"fn_greet","parameters"',
        '{"name":"fn_greet","parameters":',
        '{"name":"fn_greet","parameters":{',
        '{"name":"fn_greet","parameters":{"na',
        '{"name":"fn_greet","parameters":{"name"',
        '{"name":"fn_greet","parameters":{"name":',
        '{"name":"fn_greet","parameters":{"name":"un',
    ],
)
def test_a_growing_answer_stays_a_valid_prefix(
    text: str, definitions: FunctionDefinitionList
) -> None:
    assert JSONGrammar.valid_prefix(text, definitions)
    assert not JSONGrammar.is_closed(text, definitions)


@pytest.mark.parametrize(
    "text",
    [
        # a fragment that can never grow into what belongs here
        '{"namX',
        '{"name": "fn_zz',
        '{"name":"fn_greet","parameters":{"zz',
        # a value of the wrong type may not even begin
        '{"name":"fn_greet","parameters":{"name":tru',
        '{"name":"fn_greet","parameters":{"name":1',
        '{"name":"fn_add","parameters":{"a":"x',
        # a number fragment that could only become a float, for an
        # integer parameter
        '{"name":"fn_count","parameters":{"n":12.',
        '{"name":"fn_count","parameters":{"n":1e',
        # a fragment only counts while its string really is open, so a
        # bare word is never read as a half-written one
        '{"name":fn_greet',
        '{"name"fn_greet',
    ],
)
def test_a_fragment_that_cannot_grow_is_turned_down(
    text: str, definitions: FunctionDefinitionList
) -> None:
    assert not JSONGrammar.valid_prefix(text, definitions)
    assert not JSONGrammar.is_closed(text, definitions)


@pytest.mark.parametrize("text", INVALID_CALLS)
def test_invalid_text_is_turned_down(
    text: str, definitions: FunctionDefinitionList
) -> None:
    assert not JSONGrammar.valid_prefix(text, definitions)
    assert not JSONGrammar.is_closed(text, definitions)


# --------------------------------------------------------------------------
# Parameter keys
# --------------------------------------------------------------------------


def test_a_declared_parameter_is_accepted(
    definitions: FunctionDefinitionList,
) -> None:
    assert JSONGrammar.is_closed(
        '{"name":"fn_greet","parameters":{"name":"shrek"}}', definitions
    )


def test_an_undeclared_parameter_is_rejected(
    definitions: FunctionDefinitionList,
) -> None:
    # A key the definition does not declare is refused, so an extra
    # key can never reach the output file.
    assert not JSONGrammar.valid_prefix(
        '{"name":"fn_greet","parameters":{"name":"x","extra":1}}',
        definitions,
    )


def test_a_repeated_parameter_is_rejected(
    definitions: FunctionDefinitionList,
) -> None:
    # Using a key removes it from the outstanding set, so it cannot be
    # used twice even with a different value.
    assert not JSONGrammar.valid_prefix(
        '{"name":"fn_greet","parameters":{"name":"x","name":"y"}}',
        definitions,
    )
    assert not JSONGrammar.valid_prefix(
        '{"name":"fn_add","parameters":{"a":2,"b":3,"a":4}}', definitions
    )


def test_a_missing_parameter_is_rejected(
    definitions: FunctionDefinitionList,
) -> None:
    # `}` closes the parameters object, so it is refused while a
    # declared parameter is still outstanding.
    assert not JSONGrammar.valid_prefix(
        '{"name":"fn_add","parameters":{"a":2}}', definitions
    )
    assert not JSONGrammar.valid_prefix(
        '{"name":"fn_add","parameters":{"a":2,}}', definitions
    )
    assert not JSONGrammar.valid_prefix(
        '{"name":"fn_sub","parameters":{"regex":"b","replacement":"c"}}',
        definitions,
    )


def test_a_parameter_of_another_function_is_rejected(
    definitions: FunctionDefinitionList,
) -> None:
    # Which keys are legal depends on the function named in the first
    # key: `regex` belongs to `fn_sub`, not to `fn_greet`.
    assert not JSONGrammar.valid_prefix(
        '{"name":"fn_greet","parameters":{"regex":"b"}}', definitions
    )
    assert not JSONGrammar.valid_prefix(
        '{"name":"fn_add","parameters":{"n":1,"b":2}}', definitions
    )


@pytest.mark.parametrize(
    "keys",
    [
        '"source_string":"a","regex":"b","replacement":"c"',
        '"regex":"b","source_string":"a","replacement":"c"',
        '"replacement":"c","regex":"b","source_string":"a"',
        '"regex":"b","replacement":"c","source_string":"a"',
        '"replacement":"c","source_string":"a","regex":"b"',
    ],
)
def test_parameter_order_does_not_matter(
    keys: str, definitions: FunctionDefinitionList
) -> None:
    # The grammar tracks a set of outstanding keys, not a sequence, so
    # a valid answer does not depend on the order the definition file
    # happens to declare them in.
    text = '{"name":"fn_sub","parameters":{' + keys + "}}"
    assert JSONGrammar.is_closed(text, definitions)
    for length in range(len(text) + 1):
        assert JSONGrammar.valid_prefix(text[:length], definitions), length


# --------------------------------------------------------------------------
# Functions with no parameters, and functions with several
# --------------------------------------------------------------------------


def test_a_function_with_no_parameters_closes_on_the_empty_object(
    definitions: FunctionDefinitionList,
) -> None:
    assert JSONGrammar.is_closed(
        '{"name":"fn_now","parameters":{}}', definitions
    )
    assert not JSONGrammar.is_closed(
        '{"name":"fn_now","parameters":{', definitions
    )
    assert not JSONGrammar.valid_prefix(
        '{"name":"fn_now","parameters":{"a":1}}', definitions
    )


def test_a_function_with_two_parameters_does_not_close_after_one(
    definitions: FunctionDefinitionList,
) -> None:
    assert not JSONGrammar.is_closed(
        '{"name":"fn_add","parameters":{"a":2}}', definitions
    )
    assert JSONGrammar.is_closed(
        '{"name":"fn_add","parameters":{"a":2,"b":3}}', definitions
    )


def test_a_function_with_three_parameters_does_not_close_after_two(
    definitions: FunctionDefinitionList,
) -> None:
    two = '{"name":"fn_sub","parameters":{"source_string":"a","regex":"b"}}'
    assert not JSONGrammar.is_closed(two, definitions)
    three = (
        '{"name":"fn_sub","parameters":{"source_string":"a",'
        '"regex":"b","replacement":"c"}}'
    )
    assert JSONGrammar.is_closed(three, definitions)


def test_a_comma_is_only_wanted_while_a_parameter_is_outstanding(
    definitions: FunctionDefinitionList,
) -> None:
    # Once every declared parameter is in, the object has to close.
    assert not JSONGrammar.valid_prefix(
        '{"name":"fn_greet","parameters":{"name":"x",', definitions
    )
    assert not JSONGrammar.valid_prefix(
        '{"name":"fn_greet","parameters":{"name":"x",}}', definitions
    )


# --------------------------------------------------------------------------
# Type enforcement
# --------------------------------------------------------------------------

#: `null` is accepted for every declared type: the system prompt asks
#: for it when a required value is missing.
ACCEPTED = {
    "string": ['"shrek"', '""', '"a\\nb"', "null"],
    "number": ["2", "-3", "2.5", "-1.5e-8", "1e10", "null"],
    "integer": ["3", "-7", "0", "null"],
    # A float takes an integer too: JSON has a single number type and
    # `3` is a float. The reverse does not hold, see `integer` below.
    "float": ["3.5", "3", "1.5e-8", "-0.0", "null"],
    "boolean": ["true", "false", "null"],
}

REJECTED = {
    "string": ["5", "-3.5", "true", "false", "[1]", '{"a":1}'],
    "number": ['"2"', "true", "false", "[1]", '{"a":1}'],
    # An integer may not carry a fraction or an exponent.
    "integer": ["3.5", "1e3", "-7.0", "true", '"3"', "[3]"],
    "float": ['"3.5"', "true", "false", "[3.5]", '{"a":1.5}'],
    "boolean": ["1", "0", "3.5", '"true"', '"false"', "[true]"],
}

ACCEPTED_CASES = [
    (declared, value)
    for declared, values in sorted(ACCEPTED.items())
    for value in values
]
REJECTED_CASES = [
    (declared, value)
    for declared, values in sorted(REJECTED.items())
    for value in values
]


def call_for(declared: str, value: str) -> str:
    """The answer a one-parameter probe function would be given."""
    return '{"name":"fn_probe","parameters":{"value":' + value + "}}"


@pytest.mark.parametrize(("declared", "value"), ACCEPTED_CASES)
def test_a_value_of_the_declared_type_is_accepted(
    one_parameter: OneParameter, declared: str, value: str
) -> None:
    probe = one_parameter("fn_probe", declared)
    assert JSONGrammar.is_closed(call_for(declared, value), probe)


@pytest.mark.parametrize(("declared", "value"), REJECTED_CASES)
def test_a_value_of_the_wrong_type_is_rejected(
    one_parameter: OneParameter, declared: str, value: str
) -> None:
    probe = one_parameter("fn_probe", declared)
    text = call_for(declared, value)
    assert not JSONGrammar.valid_prefix(text, probe)
    assert not JSONGrammar.is_closed(text, probe)


def test_a_boolean_parameter_refuses_one(
    one_parameter: OneParameter,
) -> None:
    # The `bool`-is-a-subclass-of-`int` trap, one way round. A naive
    # "is this a number?" check reads `1` as a number and lets it
    # through; the tokenizer classifies with `type(parsed) in (int,
    # float)` precisely so that cannot happen.
    probe = one_parameter("fn_probe", "boolean")
    assert not JSONGrammar.is_closed(
        '{"name":"fn_probe","parameters":{"value":1}}', probe
    )
    assert not JSONGrammar.valid_prefix(
        '{"name":"fn_probe","parameters":{"value":0}}', probe
    )


def test_an_integer_parameter_refuses_true(
    one_parameter: OneParameter,
) -> None:
    # The same trap, the other way round.
    probe = one_parameter("fn_probe", "integer")
    assert not JSONGrammar.is_closed(
        '{"name":"fn_probe","parameters":{"value":true}}', probe
    )
    assert not JSONGrammar.valid_prefix(
        '{"name":"fn_probe","parameters":{"value":false}}', probe
    )


def test_a_number_parameter_refuses_a_quoted_value(
    definitions: FunctionDefinitionList,
) -> None:
    # A quoted value can only ever grow into a string, so the fragment
    # is refused as soon as its opening quote is there.
    assert not JSONGrammar.valid_prefix(
        '{"name":"fn_add","parameters":{"a":"', definitions
    )
    assert not JSONGrammar.is_closed(
        '{"name":"fn_add","parameters":{"a":"2","b":3}}', definitions
    )


def test_a_number_fragment_is_accepted_while_it_grows(
    definitions: FunctionDefinitionList,
) -> None:
    for text in (
        '{"name":"fn_add","parameters":{"a":-',
        '{"name":"fn_add","parameters":{"a":1',
        '{"name":"fn_add","parameters":{"a":12.',
        '{"name":"fn_add","parameters":{"a":1e',
        '{"name":"fn_add","parameters":{"a":1e+',
    ):
        assert JSONGrammar.valid_prefix(text, definitions), text


def test_a_boolean_fragment_is_accepted_while_it_grows(
    definitions: FunctionDefinitionList,
) -> None:
    # `t` cannot be completed into a number, but it can become `true`.
    assert JSONGrammar.valid_prefix(
        '{"name":"fn_flag","parameters":{"on":t', definitions
    )


def test_a_null_fragment_is_accepted_while_it_grows(
    definitions: FunctionDefinitionList,
) -> None:
    # `n`, `nu`, `nul` have to stay legal, or the first character of
    # the only value every type allows is turned down and the decoder
    # deadlocks on a state it is meant to be able to leave.
    for text in (
        '{"name":"fn_add","parameters":{"a":n',
        '{"name":"fn_add","parameters":{"a":nu',
        '{"name":"fn_add","parameters":{"a":nul',
    ):
        assert JSONGrammar.valid_prefix(text, definitions), text


# --------------------------------------------------------------------------
# Nesting: what the grammar does today
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "key", "refused"),
    [
        ("fn_items", "items", ("[]", "[1]", "[")),
        ("fn_config", "cfg", ("{}", '{"a":1}', "{")),
        ("fn_unknown", "x", ("5", '"text"', "true", "[]")),
    ],
)
def test_a_nested_or_unknown_type_takes_null_only(
    definitions: FunctionDefinitionList,
    name: str,
    key: str,
    refused: tuple[str, ...],
) -> None:
    # Documenting the current behaviour, not the wanted behaviour. The
    # walk never enters a nested value, so the only token an `array`, an
    # `object` or an unrecognised type accepts is `null`: offering `[`
    # or `{` would be accepted and then dead-end on the very next
    # token, with no way back out of the nested structure.
    head = '{"name":"' + name + '","parameters":{"' + key + '":'
    assert JSONGrammar.is_closed(head + "null}}", definitions)
    for value in refused:
        assert not JSONGrammar.valid_prefix(head + value + "}}", definitions)
        assert not JSONGrammar.is_closed(head + value + "}}", definitions)


# --------------------------------------------------------------------------
# The invariant the whole design rests on
# --------------------------------------------------------------------------


def test_the_grammar_closes_exactly_the_valid_calls(
    definitions: FunctionDefinitionList,
) -> None:
    corpus = VALID_CALLS + INVALID_CALLS + NOT_JSON
    closed = [
        text for text in corpus if JSONGrammar.is_closed(text, definitions)
    ]
    assert closed == VALID_CALLS


@pytest.mark.parametrize(
    "text", MALFORMED, ids=[r for r, _ in MALFORMED_CASES]
)
def test_malformed_json_is_never_certified_as_closed(
    text: str, definitions: FunctionDefinitionList
) -> None:
    # A real bug, fixed: `"a\d"` was read as a closed string, so the
    # grammar certified a finished call that `json.loads` then
    # rejected. The tokenizer now raises on an escape JSON does not
    # define and the walk turns that into "not a valid prefix" — not
    # just "not closed", because there is nothing this text could grow
    # into.
    with pytest.raises(json.JSONDecodeError):
        json.loads(text)
    assert not JSONGrammar.is_closed(text, definitions)
    assert not JSONGrammar.valid_prefix(text, definitions)


@pytest.mark.parametrize(
    "text", UNTERMINATED, ids=[r for r, _ in UNTERMINATED_CASES]
)
def test_unterminated_json_is_never_closed_but_is_a_valid_prefix(
    text: str, definitions: FunctionDefinitionList
) -> None:
    # The other half of the same idea: text that is not JSON is not
    # automatically wrong. An answer that stops part way through is
    # exactly what the decoder sees at every step but the last, so it
    # has to be accepted as a prefix and refused as a finished call.
    with pytest.raises(json.JSONDecodeError):
        json.loads(text)
    assert JSONGrammar.valid_prefix(text, definitions)
    assert not JSONGrammar.is_closed(text, definitions)


@pytest.mark.parametrize("text", VALID_CALLS)
def test_everything_the_grammar_closes_can_be_read_back_as_json(
    text: str, definitions: FunctionDefinitionList
) -> None:
    assert JSONGrammar.is_closed(text, definitions)
    json.loads(text)  # raises if the grammar certified bad JSON


@pytest.mark.parametrize("text", VALID_JSON_NOT_A_CALL)
def test_well_formed_json_is_not_enough_to_be_a_call(
    text: str, definitions: FunctionDefinitionList
) -> None:
    # The other direction: a parsable object is necessary, not
    # sufficient. Without this, a grammar that simply accepted every
    # JSON object would pass every other test here.
    json.loads(text)
    assert not JSONGrammar.is_closed(text, definitions)
