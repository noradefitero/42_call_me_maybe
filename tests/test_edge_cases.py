"""The edge cases `subject.pdf` calls out, one group per case.

| Case | Where it is below |
|---|---|
| Empty string values | `test_an_empty_string_value` |
| Very large numbers | `test_a_very_large_number` |
| Special characters | `test_a_special_character_in_a_value` |
| Wrong types | `test_a_value_of_the_wrong_type` |
| Ambiguous prompts | `test_an_ambiguous_prompt_*` |
| Multiple parameters | `test_a_function_with_several_parameters` |

No test here loads the model. The ambiguous-prompt case is really a
statement about model *behaviour*, which no unit test can pin down, so
what is asserted is the part that is this project's job: the answer
stays valid whichever function the model picks, the prompt reaches the
generator byte for byte, and the run produces one output object per
input prompt.
"""

import json

import pytest

from call_me_maybe.config import (
    DEFAULT_FUNCTIONS_FILE,
    DEFAULT_INPUT_FILE,
)
from call_me_maybe.json_grammar import JSONGrammar
from call_me_maybe.json_parser import JSONTokenizer
from call_me_maybe.models.function_call import FunctionCall
from call_me_maybe.models.function_lists import (
    FunctionCallList,
    FunctionDefinitionList,
)
from call_me_maybe.models.llm_function_call import LLMFunctionCall
from call_me_maybe.pipeline.parse import run
from call_me_maybe.prompt import Prompt

# A single backslash, built rather than typed, so the JSON escape cases
# below are unambiguous next to the ordinary Python escapes around them.
BS = chr(92)


# --------------------------------------------------------------------------
# Empty string values
# --------------------------------------------------------------------------


def test_an_empty_string_value(
    definitions: FunctionDefinitionList,
) -> None:
    # `""` is a complete string token, not an unfinished one: the
    # closing quote is there, so nothing about it is still growing.
    assert JSONTokenizer.tokenize('""') != []
    text = '{"name":"fn_greet","parameters":{"name":""}}'
    assert JSONTokenizer.tokenize('""')[0].value == ""
    assert JSONGrammar.is_closed(text, definitions)
    assert json.loads(text)["parameters"] == {"name": ""}


def test_an_empty_string_does_not_hide_a_missing_value(
    definitions: FunctionDefinitionList,
) -> None:
    # An empty value and an absent key are different things, and the
    # grammar keeps them apart.
    assert JSONGrammar.is_closed(
        '{"name":"fn_greet","parameters":{"name":""}}', definitions
    )
    assert not JSONGrammar.valid_prefix(
        '{"name":"fn_greet","parameters":{}}', definitions
    )


def test_an_empty_string_in_every_position(
    definitions: FunctionDefinitionList,
) -> None:
    text = (
        '{"name":"fn_sub","parameters":{"source_string":"",'
        '"regex":"","replacement":""}}'
    )
    assert JSONGrammar.is_closed(text, definitions)
    assert json.loads(text)["parameters"] == {
        "source_string": "",
        "regex": "",
        "replacement": "",
    }


# --------------------------------------------------------------------------
# Very large numbers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "number",
    [
        "0",
        "-0",
        "9007199254740991",
        "9007199254740993",
        "1234567890123456789012345678901234567890",
        "1e10",
        "1.7976931348623157e308",
        "-1.5e-300",
        "2.5",
        "1e-8",
    ],
)
def test_a_number_is_judged_by_its_token_type_not_its_size(
    number: str, definitions: FunctionDefinitionList
) -> None:
    # The grammar's number rule is "the token is a JSON number", so
    # magnitude is irrelevant to validity: a value too big for a float
    # is exactly as acceptable as `2`.
    text = '{"name":"fn_add","parameters":{"a":' + number + ',"b":0}}'
    assert JSONTokenizer.tokenize(number)[0].type.name == "Number"
    assert JSONGrammar.is_closed(text, definitions)
    assert json.loads(text)["parameters"]["a"] is not None


def test_a_very_large_number_reaches_the_output_unchanged(
    definitions: FunctionDefinitionList,
) -> None:
    # Pydantic is the last boundary, so the digits have to survive it
    # rather than being rounded on the way through.
    big = 9007199254740993
    text = '{"name":"fn_add","parameters":{"a":' + str(big) + ',"b":1}}'
    call = LLMFunctionCall.model_validate_json(text)
    assert call.parameters["a"] == big
    assert call.name == "fn_add"


def test_a_leading_zero_is_refused(
    definitions: FunctionDefinitionList,
) -> None:
    # `0` is a number and `01` is not valid JSON, so the digit after a
    # leading zero is a token the grammar has to turn down.
    assert JSONGrammar.is_closed(
        '{"name":"fn_add","parameters":{"a":0,"b":1}}', definitions
    )
    assert not JSONGrammar.valid_prefix(
        '{"name":"fn_add","parameters":{"a":01,"b":1}}', definitions
    )


# --------------------------------------------------------------------------
# Special characters in values
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # nothing to decode
        ('"!@#$%^&*()"', "!@#$%^&*()"),
        ('"<script>"', "<script>"),
        ('"100%"', "100%"),
        ('"café"', "café"),
        ('"it\'s"', "it's"),
        ('""', ""),
        # an escape, decoded on the way back out
        ('"a\\"b"', 'a"b'),
        ('"a\\\\b"', "a" + BS + "b"),
        ('"a\\nb"', "a\nb"),
        ('"a\\tb"', "a\tb"),
        ('"a\\u0041b"', "aAb"),
        ('"a\\u00e9"', "aé"),
        ('"a\\/b"', "a/b"),
    ],
)
def test_a_special_character_in_a_value(
    value: str,
    expected: str,
    definitions: FunctionDefinitionList,
) -> None:
    # `json.loads` is the arbiter of what a special character is
    # allowed to be. The grammar only sees the raw span between the
    # quotes, so the answer being valid JSON here is what proves the
    # escape was read as content and not as the end of the string.
    text = '{"name":"fn_greet","parameters":{"name":' + value + "}}"
    assert JSONGrammar.is_closed(text, definitions)
    assert json.loads(text)["parameters"]["name"] == expected


def test_a_quoted_phrase_inside_a_value_stays_one_value(
    definitions: FunctionDefinitionList,
) -> None:
    # The case from the shipped dataset: the model emitted a value with
    # quotes inside it. Before `__find_closing_quote` existed, the scan
    # stopped at the first one, read a truncated string, and the run
    # deadlocked on an answer it could never close.
    text = (
        '{"name":"fn_sub","parameters":{"source_string":'
        '"Hello 34 I\'m 233 years old",'
        '"regex":"[0-9]+","replacement":"NUMBERS"}}'
    )
    assert JSONGrammar.is_closed(text, definitions)
    call = LLMFunctionCall.model_validate_json(text)
    assert call.parameters["source_string"] == "Hello 34 I'm 233 years old"


def test_a_regex_with_backslashes_survives_the_round_trip(
    definitions: FunctionDefinitionList,
) -> None:
    text = (
        '{"name":"fn_sub","parameters":{"source_string":"x",'
        '"regex":"' + BS + BS + 'd+","replacement":"N"}}'
    )
    assert JSONGrammar.is_closed(text, definitions)
    assert json.loads(text)["parameters"]["regex"] == BS + "d+"


@pytest.mark.parametrize("escape", ["d", "q", "x", "z"])
def test_a_special_character_that_is_not_a_json_escape_is_refused(
    escape: str, definitions: FunctionDefinitionList
) -> None:
    text = '{"name":"fn_greet","parameters":{"name":"a' + BS + escape + 'b"}}'
    with pytest.raises(ValueError, match="Invalid escape sequence"):
        JSONTokenizer.tokenize(text)
    assert not JSONGrammar.is_closed(text, definitions)


# --------------------------------------------------------------------------
# Wrong types
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        '{"name":"fn_greet","parameters":{"name":5}}',
        '{"name":"fn_greet","parameters":{"name":true}}',
        '{"name":"fn_greet","parameters":{"name":[1]}}',
        '{"name":"fn_greet","parameters":{"name":{"a":1}}}',
        '{"name":"fn_add","parameters":{"a":"2","b":3}}',
        '{"name":"fn_add","parameters":{"a":true,"b":3}}',
        '{"name":"fn_flag","parameters":{"on":1}}',
        '{"name":"fn_count","parameters":{"n":3.5}}',
        '{"name":"fn_count","parameters":{"n":true}}',
    ],
)
def test_a_value_of_the_wrong_type(
    text: str, definitions: FunctionDefinitionList
) -> None:
    # The decoder cannot emit any of these, so they cannot appear in
    # the output file even though every one of them is good JSON.
    json.loads(text)
    assert not JSONGrammar.valid_prefix(text, definitions)
    assert not JSONGrammar.is_closed(text, definitions)


# --------------------------------------------------------------------------
# Several parameters
# --------------------------------------------------------------------------


def test_a_function_with_several_parameters(
    definitions: FunctionDefinitionList,
) -> None:
    text = (
        '{"name":"fn_sub","parameters":{"source_string":"The cat sat",'
        '"regex":"cat","replacement":"dog"}}'
    )
    assert JSONGrammar.is_closed(text, definitions)
    call = LLMFunctionCall.model_validate_json(text)
    assert call.parameters == {
        "source_string": "The cat sat",
        "regex": "cat",
        "replacement": "dog",
    }


def test_several_parameters_in_any_order(
    definitions: FunctionDefinitionList,
) -> None:
    text = (
        '{"name":"fn_sub","parameters":{"replacement":"dog",'
        '"source_string":"The cat sat","regex":"cat"}}'
    )
    assert JSONGrammar.is_closed(text, definitions)
    # Reordering the keys changes nothing about the call.
    assert json.loads(text) == {
        "name": "fn_sub",
        "parameters": {
            "replacement": "dog",
            "source_string": "The cat sat",
            "regex": "cat",
        },
    }


def test_one_parameter_of_several_missing_is_refused(
    definitions: FunctionDefinitionList,
) -> None:
    text = '{"name":"fn_sub","parameters":{"source_string":"a","regex":"b"}}'
    assert not JSONGrammar.is_closed(text, definitions)


# --------------------------------------------------------------------------
# Ambiguous prompts
# --------------------------------------------------------------------------

#: One prompt, two functions it could reasonably mean.
AMBIGUOUS_PROMPT = "What is the sum of 2 and 3?"

AMBIGUOUS_ANSWERS = [
    '{"name":"fn_add_numbers","parameters":{"a":2,"b":3}}',
    '{"name":"fn_greet","parameters":{"name":"2 + 3"}}',
]

#: One closed answer per prompt of the shipped dataset, standing in for
#: what the model produced on the recorded run. The order matches
#: `data/input/function_calling_tests.json`.
DATASET_ANSWERS = [
    '{"name":"fn_add_numbers","parameters":{"a":2,"b":3}}',
    '{"name":"fn_add_numbers","parameters":{"a":265,"b":345}}',
    '{"name":"fn_greet","parameters":{"name":"shrek"}}',
    '{"name":"fn_greet","parameters":{"name":"john"}}',
    '{"name":"fn_reverse_string","parameters":{"s":"hello"}}',
    '{"name":"fn_reverse_string","parameters":{"s":"world"}}',
    '{"name":"fn_get_square_root","parameters":{"a":16}}',
    '{"name":"fn_get_square_root","parameters":{"a":144}}',
    (
        '{"name":"fn_substitute_string_with_regex","parameters":'
        '{"source_string":"Hello 34 I\'m 233 years old",'
        '"regex":"[0-9]+","replacement":"NUMBERS"}}'
    ),
    (
        '{"name":"fn_substitute_string_with_regex","parameters":'
        '{"source_string":"Programming is fun","regex":"[aeiou]",'
        '"replacement":"*"}}'
    ),
    (
        '{"name":"fn_substitute_string_with_regex","parameters":'
        '{"source_string":"The cat sat on the mat with another cat",'
        '"regex":"cat","replacement":"dog"}}'
    ),
]


def answer_for(prompt_text: str, answer: str) -> FunctionCall:
    """Stand in for the model: validate an answer, add the prompt back.

    This is the last half of `pipeline/runner.py` with the generation
    loop removed: pydantic confirms the answer, the prompt goes back
    on, and the result is what the output file is built from.
    """
    call = LLMFunctionCall.model_validate_json(answer)
    return FunctionCall(
        prompt=prompt_text,
        name=call.name,
        parameters=call.parameters,
    )


def test_an_ambiguous_prompt_stays_valid_whichever_function_is_picked() -> (
    None
):
    # Which function the model chooses is its business; the guarantee
    # this project makes is that either answer is a well-formed call.
    # Nothing here is about which one is *right*.
    definitions = FunctionDefinitionList.model_validate_json(
        DEFAULT_FUNCTIONS_FILE.read_text()
    )
    for answer in AMBIGUOUS_ANSWERS:
        assert JSONGrammar.is_closed(answer, definitions)
        call = LLMFunctionCall.model_validate_json(answer)
        assert call.name in {f.name for f in definitions.root}


def test_the_prompt_reaches_the_generator_verbatim() -> None:
    # The prompt is substituted into the template untouched: a prompt
    # that is ambiguous, or that contains braces, quotes or dollars,
    # must not be reshaped on its way to the model.
    prompt = Prompt(
        template="<|im_start|>user\n$prompt<|im_end|>",
        system="sys",
        user_prompt=AMBIGUOUS_PROMPT,
    )
    assert prompt.user_prompt == AMBIGUOUS_PROMPT
    assert str(prompt).endswith(AMBIGUOUS_PROMPT + "<|im_end|>")


def test_a_prompt_with_punctuation_is_not_rewritten() -> None:
    tricky = "Sum {2} + $3? \"quoted\" \\ and 'single'"
    prompt = Prompt(
        template="U=$prompt",
        user_prompt=tricky,
    )
    assert prompt.user_prompt == tricky
    assert str(prompt) == "U=" + tricky


def test_one_output_object_per_input_prompt() -> None:
    # The output file has one entry per input, in the same order, with
    # the prompt put back. Driven end to end over the shipped dataset
    # with the answer the grammar accepts, so no model is involved.
    parsed = run(None, None, DEFAULT_FUNCTIONS_FILE, DEFAULT_INPUT_FILE)
    prompts = parsed["input"]
    assert prompts
    results = [
        answer_for(prompt_text, answer)
        for prompt_text, answer in zip(prompts, DATASET_ANSWERS)
    ]
    assert len(results) == len(prompts)
    dumped = json.loads(FunctionCallList(results).model_dump_json(indent=4))
    assert [entry["prompt"] for entry in dumped] == prompts
    for entry in dumped:
        assert sorted(entry) == ["name", "parameters", "prompt"]


def test_the_shipped_dataset_is_read_unchanged() -> None:
    parsed = run(None, None, DEFAULT_FUNCTIONS_FILE, DEFAULT_INPUT_FILE)
    on_disk = json.loads(DEFAULT_INPUT_FILE.read_text())
    assert parsed["input"] == [entry["prompt"] for entry in on_disk]
    assert len(parsed["definitions"].root) == len(
        json.loads(DEFAULT_FUNCTIONS_FILE.read_text())
    )


@pytest.mark.parametrize("answer", DATASET_ANSWERS)
def test_every_stored_answer_is_a_closed_call(answer: str) -> None:
    # The stand-in answers are checked by the same grammar the decoder
    # uses, so the plumbing test above is testing real answers.
    definitions = FunctionDefinitionList.model_validate_json(
        DEFAULT_FUNCTIONS_FILE.read_text()
    )
    assert JSONGrammar.is_closed(answer, definitions), answer
