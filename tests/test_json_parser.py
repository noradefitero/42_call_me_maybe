"""Unit tests for the tokenizer: text in, the expected token list out.

`json_parser.py` is a pure function, so nothing here needs the model.
The property the grammar leans on is the one split into `Incomplete`:
a trailing word that could still grow into a valid token is reported
as `Incomplete`, and one that never could is an error.
"""

import pytest

from call_me_maybe.json_parser import (
    JSONToken,
    JSONTokenizer,
    JSONTokenType,
)

# A single backslash, built rather than typed, so the JSON escape cases
# below are unambiguous next to the ordinary Python escapes around them.
BS = chr(92)

#: The escapes JSON itself defines, and the escapes it does not.
VALID_ESCAPES = ("n", "t", "r", "b", "f", "u")
INVALID_ESCAPES = ("d", "q", "x", "z", "'")


# --------------------------------------------------------------------------
# Punctuation and structure
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("{", JSONTokenType.BraceOpen),
        ("}", JSONTokenType.BraceClose),
        ("[", JSONTokenType.BracketOpen),
        ("]", JSONTokenType.BracketClose),
        (":", JSONTokenType.Colon),
        (",", JSONTokenType.Comma),
    ],
)
def test_each_punctuation_character_is_one_token(
    text: str, expected: JSONTokenType
) -> None:
    assert JSONTokenizer.tokenize(text) == [JSONToken(expected, text)]


def test_object_is_split_into_punctuation_strings_and_values() -> None:
    assert JSONTokenizer.tokenize('{"a":1}') == [
        JSONToken(JSONTokenType.BraceOpen, "{"),
        JSONToken(JSONTokenType.String, "a"),
        JSONToken(JSONTokenType.Colon, ":"),
        JSONToken(JSONTokenType.Number, "1"),
        JSONToken(JSONTokenType.BraceClose, "}"),
    ]


def test_array_is_split_into_brackets_values_and_commas() -> None:
    assert JSONTokenizer.tokenize("[1,2]") == [
        JSONToken(JSONTokenType.BracketOpen, "["),
        JSONToken(JSONTokenType.Number, "1"),
        JSONToken(JSONTokenType.Comma, ","),
        JSONToken(JSONTokenType.Number, "2"),
        JSONToken(JSONTokenType.BracketClose, "]"),
    ]


def test_a_whole_call_is_split_in_order() -> None:
    assert JSONTokenizer.tokenize('{"name":"f","parameters":{}}') == [
        JSONToken(JSONTokenType.BraceOpen, "{"),
        JSONToken(JSONTokenType.String, "name"),
        JSONToken(JSONTokenType.Colon, ":"),
        JSONToken(JSONTokenType.String, "f"),
        JSONToken(JSONTokenType.Comma, ","),
        JSONToken(JSONTokenType.String, "parameters"),
        JSONToken(JSONTokenType.Colon, ":"),
        JSONToken(JSONTokenType.BraceOpen, "{"),
        JSONToken(JSONTokenType.BraceClose, "}"),
        JSONToken(JSONTokenType.BraceClose, "}"),
    ]


@pytest.mark.parametrize("text", ["", " ", "\n", "\t", "  \r\n\t "])
def test_whitespace_only_text_yields_no_tokens(text: str) -> None:
    assert JSONTokenizer.tokenize(text) == []


def test_whitespace_between_tokens_is_dropped() -> None:
    assert JSONTokenizer.tokenize(' { "a" : \n 1 } ') == [
        JSONToken(JSONTokenType.BraceOpen, "{"),
        JSONToken(JSONTokenType.String, "a"),
        JSONToken(JSONTokenType.Colon, ":"),
        JSONToken(JSONTokenType.Number, "1"),
        JSONToken(JSONTokenType.BraceClose, "}"),
    ]


def test_an_unexpected_character_is_an_error() -> None:
    with pytest.raises(ValueError, match="Unexpected character"):
        JSONTokenizer.tokenize("{" + "+" + "}")


# --------------------------------------------------------------------------
# Complete strings
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ('"abc"', "abc"),
        ('""', ""),
        ('"a b"', "a b"),
        ('"  padded  "', "  padded  "),
        ('"café"', "café"),
        ('"3 + 4 = 7"', "3 + 4 = 7"),
        ('"!@#$%^&*()"', "!@#$%^&*()"),
    ],
)
def test_a_closed_string_is_one_token_with_its_content(
    text: str, value: str
) -> None:
    assert JSONTokenizer.tokenize(text) == [
        JSONToken(JSONTokenType.String, value)
    ]


def test_the_tokenizer_does_not_unescape_a_string() -> None:
    # The value is the raw span between the quotes, escapes included:
    # the grammar never needs the decoded text, and decoding here
    # would double the work on the hot path.
    assert JSONTokenizer.tokenize(r'"a\nb"') == [
        JSONToken(JSONTokenType.String, r"a\nb")
    ]


# --------------------------------------------------------------------------
# Escape sequences
# --------------------------------------------------------------------------


@pytest.mark.parametrize("escape", VALID_ESCAPES)
def test_every_escape_json_defines_stays_inside_the_string(
    escape: str,
) -> None:
    # `\u` needs four hex digits, so the other five are one char.
    payload = "0041" if escape == "u" else escape
    text = '"a' + BS + escape + payload + 'b"'
    assert JSONTokenizer.tokenize(text) == [
        JSONToken(JSONTokenType.String, "a" + BS + escape + payload + "b")
    ]


@pytest.mark.parametrize("escape", INVALID_ESCAPES)
def test_an_escape_json_does_not_define_is_rejected(escape: str) -> None:
    # A real bug, fixed: the string scan used to stop at the first
    # quote whatever stood before it, so `"a\db"` was read as the
    # closed string `a\d` followed by garbage. The grammar then read a
    # truncated answer it could never close.
    with pytest.raises(ValueError, match="Invalid escape sequence"):
        JSONTokenizer.tokenize('"a' + BS + escape + 'b"')


def test_a_short_unicode_escape_is_rejected() -> None:
    # Only two hex digits before the closing quote, so the escape can
    # never be completed: this is a value cut off mid-generation, not
    # a valid `A`.
    with pytest.raises(ValueError, match="Invalid unicode escape"):
        JSONTokenizer.tokenize('"a' + BS + 'u041"')


def test_a_unicode_escape_with_bad_hex_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid unicode escape"):
        JSONTokenizer.tokenize('"a' + BS + 'uZZZZb"')


def test_an_escaped_quote_does_not_close_the_string() -> None:
    # `a\"b` is content, so the string is still open afterwards.
    assert JSONTokenizer.tokenize(r'"a\"b"') == [
        JSONToken(JSONTokenType.String, r"a\"b")
    ]


def test_an_escaped_backslash_keeps_the_next_quote_unescaped() -> None:
    # `\\` is one escaped backslash, so the quote after it really does
    # close the string.
    assert JSONTokenizer.tokenize(r'"a\\"') == [
        JSONToken(JSONTokenType.String, r"a\\")
    ]


def test_a_backslash_after_the_closing_quote_is_left_alone() -> None:
    # The escape search is bounded to the string's own span: the
    # backslash inside the second string cannot escape the quote that
    # closed the first one.
    assert JSONTokenizer.tokenize(r'"a" "b\n"') == [
        JSONToken(JSONTokenType.String, "a"),
        JSONToken(JSONTokenType.String, r"b\n"),
    ]


def test_a_trailing_backslash_leaves_the_string_incomplete() -> None:
    tokens = JSONTokenizer.tokenize('"a' + BS)
    assert tokens == [
        JSONToken(JSONTokenType.Incomplete, "a" + BS),
    ]


# --------------------------------------------------------------------------
# `Incomplete`: the unterminated string
# --------------------------------------------------------------------------


def test_an_unterminated_string_is_incomplete_from_its_start() -> None:
    # The central property of `Incomplete`. The value is the content
    # from the string's own opening quote to the end of the text: the
    # fragment is anchored at the START of the string, not at the end.
    text = '{"a": "unterminated string here'
    tokens = JSONTokenizer.tokenize(text)
    assert tokens == [
        JSONToken(JSONTokenType.BraceOpen, "{"),
        JSONToken(JSONTokenType.String, "a"),
        JSONToken(JSONTokenType.Colon, ":"),
        JSONToken(JSONTokenType.Incomplete, "unterminated string here"),
    ]


@pytest.mark.parametrize(
    "text",
    [
        '"',
        '"a',
        '"abcdef',
        '{"name": "fn_ad',
        '{"name":"fn_add","parameters":{"a":"12',
    ],
)
def test_an_unterminated_string_reports_everything_after_the_quote(
    text: str,
) -> None:
    quote = text.rindex('"')
    tokens = JSONTokenizer.tokenize(text)
    assert tokens[-1] == JSONToken(JSONTokenType.Incomplete, text[quote + 1 :])


def test_incomplete_only_ever_appears_last() -> None:
    # The grammar only accepts a fragment as the final token, so a
    # tokenizer that emitted one in the middle would be unusable.
    texts = [
        '"a" "b',
        '"a" tr',
        '{"name": "fn_ad',
        '"a" 1e+',
    ]
    for text in texts:
        tokens = JSONTokenizer.tokenize(text)
        incomplete = [
            index
            for index, token in enumerate(tokens)
            if token.type is JSONTokenType.Incomplete
        ]
        assert incomplete in ([], [len(tokens) - 1]), text


# --------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "0",
        "7",
        "12",
        "-12",
        "12.5",
        "-12.5",
        "1e3",
        "1E3",
        "1.5e-3",
        "-12.5e3",
        "1e+10",
    ],
)
def test_a_complete_number_is_one_number_token(text: str) -> None:
    assert JSONTokenizer.tokenize(text) == [
        JSONToken(JSONTokenType.Number, text)
    ]


@pytest.mark.parametrize("text", ["-", "12.", "1e", "1e+", "1e-", "1.5e-"])
def test_a_growable_number_fragment_is_incomplete(text: str) -> None:
    # Reported as `Incomplete` so the decoder can keep generating: one
    # more digit completes the token.
    assert JSONTokenizer.tokenize(text) == [
        JSONToken(JSONTokenType.Incomplete, text)
    ]


@pytest.mark.parametrize("text", ["-0", "0.0", "-0.0", "0"])
def test_a_number_that_looks_unfinished_is_still_a_number(text: str) -> None:
    # `0` and `-0` do not end in a dangling sign or separator, so they
    # are finished numbers and not fragments.
    assert JSONTokenizer.tokenize(text) == [
        JSONToken(JSONTokenType.Number, text)
    ]


@pytest.mark.parametrize("text", ["01", "6.e", "-a", "1.2.3", "-.5.5"])
def test_something_that_is_never_a_number_is_an_error(text: str) -> None:
    with pytest.raises(ValueError):
        JSONTokenizer.tokenize(text)


def test_a_leading_zero_is_not_a_number() -> None:
    # `0` is a number and `01` is not, so the digit after a leading
    # zero is what the grammar has to turn down.
    assert JSONTokenizer.tokenize("0") == [
        JSONToken(JSONTokenType.Number, "0")
    ]
    with pytest.raises(ValueError, match="Unexpected value"):
        JSONTokenizer.tokenize("01")


def test_a_very_large_number_is_one_token() -> None:
    big = "9" * 40
    assert JSONTokenizer.tokenize(big) == [
        JSONToken(JSONTokenType.Number, big)
    ]


# --------------------------------------------------------------------------
# Literals
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("true", JSONTokenType.BoolTrue),
        ("false", JSONTokenType.BoolFalse),
        ("null", JSONTokenType.Null),
    ],
)
def test_a_literal_is_its_own_token_type(
    text: str, expected: JSONTokenType
) -> None:
    assert JSONTokenizer.tokenize(text) == [JSONToken(expected, text)]


@pytest.mark.parametrize(
    "text",
    ["t", "tr", "tru", "f", "fa", "fal", "fals", "n", "nu", "nul"],
)
def test_a_keyword_prefix_is_incomplete(text: str) -> None:
    # `n`, `nu`, `nul` have to be accepted while they are being typed,
    # otherwise the very first character of `null` is refused and the
    # decoder deadlocks on a value it is allowed to produce.
    assert JSONTokenizer.tokenize(text) == [
        JSONToken(JSONTokenType.Incomplete, text)
    ]


@pytest.mark.parametrize("text", ["hell", "hello", "nul1", "trueish"])
def test_a_word_that_is_never_a_literal_is_an_error(text: str) -> None:
    with pytest.raises(ValueError, match="Unexpected value"):
        JSONTokenizer.tokenize(text)


# --------------------------------------------------------------------------
# Idempotence
# --------------------------------------------------------------------------

CORPUS = [
    "",
    "   ",
    "{",
    '{"name":"fn_greet","parameters":{"name":"shrek"}}',
    '{"name":"fn_greet","parameters":{"name":""}}',
    r'{"name":"fn_greet","parameters":{"name":"He said \"hi\""}}',
    r'{"name":"fn_greet","parameters":{"name":"a\\b"}}',
    r'{"name":"fn_sub","parameters":{"s":"x","regex":"\\d+","r":"N"}}',
    '{"name":"fn_add","parameters":{"a":-7,"b":3}}',
    '{"name":"fn_add","parameters":{"a":1e10,"b":9007199254740993}}',
    '{"name":"fn_flag","parameters":{"on":true}}',
    '{"name":"fn_flag","parameters":{"on":null}}',
    '{"name": "fn_ad',
    '{"name": "fn_gr',
    '{"name":"fn_greet","parameters":{"na',
    '{"name":"fn_count","parameters":{"n":-',
    '{"name":"fn_count","parameters":{"n":1e',
    "tru",
    "nul",
    "12.",
]


@pytest.mark.parametrize("text", CORPUS)
def test_tokenizing_twice_gives_the_same_tokens(text: str) -> None:
    # `valid_prefix` re-reads the whole answer on every candidate
    # token, so the same text is tokenized many times over. A scan
    # that depended on earlier calls would make the grammar
    # non-deterministic.
    assert JSONTokenizer.tokenize(text) == JSONTokenizer.tokenize(text)


@pytest.mark.parametrize("text", CORPUS)
def test_a_rejected_text_stays_rejected(text: str) -> None:
    try:
        JSONTokenizer.tokenize(text)
    except ValueError:
        return
    assert JSONTokenizer.tokenize(text) == JSONTokenizer.tokenize(text)
