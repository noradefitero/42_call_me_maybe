import json
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar


class JSONTokenType(Enum):
    """The kind of token `JSONTokenizer` reads out of a JSON text."""

    BraceOpen = 0
    BraceClose = 1
    BracketOpen = 2
    BracketClose = 3
    String = 4
    Number = 5
    Comma = 6
    Colon = 7
    BoolTrue = 8
    BoolFalse = 9
    Null = 10
    Incomplete = 11


@dataclass(frozen=True)
class JSONToken:
    """A token: its `type` and the raw `value` read for it."""

    type: JSONTokenType
    value: str


class JSONTokenizer:
    """Reads a partial JSON text into `JSONToken`s as it arrives."""

    _LITERALS: ClassVar[dict[str, JSONTokenType]] = {
        "true": JSONTokenType.BoolTrue,
        "false": JSONTokenType.BoolFalse,
        "null": JSONTokenType.Null,
    }
    _PUNCTUATION: ClassVar[dict[str, JSONTokenType]] = {
        "{": JSONTokenType.BraceOpen,
        "}": JSONTokenType.BraceClose,
        "[": JSONTokenType.BracketOpen,
        "]": JSONTokenType.BracketClose,
        ":": JSONTokenType.Colon,
        ",": JSONTokenType.Comma,
    }
    # What may sit inside a number: the decimal point, the exponent
    # letter and both signs. Two rules need this set, so it lives here.
    _NUMBER_CHARS: ClassVar[tuple[str, ...]] = (".", "e", "E", "+", "-")

    @staticmethod
    def __is_number(value: str) -> bool:
        """True if `value` is a complete JSON number, like `-12.5e3`."""
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return False
        # `type()` instead of `isinstance()`: bool is a subclass of int
        # and "true"/"false" are literals, not numbers.
        return type(parsed) in (int, float)

    @staticmethod
    def __classify_number(value: str) -> JSONTokenType | None:
        """Decide the token type for a word that looks numeric.

        Number:     complete JSON number (`-12.5e3`).
        Incomplete: could still grow into one (`12.`, `1e+`, `-`).
        None:       not a number at all (`hello`, `6.e`).
        """
        if JSONTokenizer.__is_number(value):
            return JSONTokenType.Number
        # "-", "12.", "1e" or "1e+" end with a dangling sign or separator:
        # adding one digit would complete them.
        if value.endswith(JSONTokenizer._NUMBER_CHARS) and (
            JSONTokenizer.__is_number(value + "0")
        ):
            return JSONTokenType.Incomplete
        return None

    @staticmethod
    def __is_word_char(char: str, value: str) -> bool:
        """True if `char` goes on in the word `value` read so far.

        Letters, digits and "_" always do. So does a leading "-" (a
        negative number) and any ".", "e" or sign that keeps the word a
        valid number fragment.
        """
        if char.isalnum() or char == "_":
            return True
        if not value and char == "-":
            return True
        if char not in JSONTokenizer._NUMBER_CHARS:
            return False
        return JSONTokenizer.__classify_number(value + char) is not None

    @staticmethod
    def __scan_word(input: str, start: int) -> tuple[str, int]:
        """Consume one word (keyword or number) starting at `start`.

        Returns the consumed value and the index just past it.
        """
        value = ""
        current = start
        while current < len(input):
            char = input[current]
            if JSONTokenizer.__is_word_char(char, value):
                value += char
                current += 1
            else:
                break
        return value, current

    @staticmethod
    def __tokenize_word(value: str) -> JSONToken:
        literal = JSONTokenizer._LITERALS.get(value)
        if literal is not None:
            return JSONToken(literal, value)
        number_type = JSONTokenizer.__classify_number(value)
        if number_type is not None:
            return JSONToken(number_type, value)
        # A proper prefix of a keyword ("tru") could still grow into a
        # valid token.
        if any(word.startswith(value) for word in JSONTokenizer._LITERALS):
            return JSONToken(JSONTokenType.Incomplete, value)
        raise ValueError(f"Unexpected value: {value}")

    # What a backslash may protect: the escapes JSON itself defines.
    _ESCAPES: ClassVar[str] = '"\\/bfnrtu'

    @staticmethod
    def __find_closing_quote(input: str, start: int) -> int:
        """Index of the quote closing the string, or -1 if there is none.

        A backslash escapes whatever follows it, so `\"` does not end
        the string. `start` is the index just after the opening quote.
        Both scans stay in `str.find` so this stays cheap on the hot
        path, where the grammar re-reads the answer for every candidate
        token.

        Raises ValueError on an escape JSON does not define, so a
        string can never be read as closed when it is not valid JSON.
        """
        current = start
        while True:
            quote = input.find('"', current)
            if quote == -1:
                return -1
            # Bounded to this string: a backslash past the quote cannot.
            escape = input.find("\\", current, quote)
            if escape == -1:
                return quote
            escaped = input[escape + 1 : escape + 2]
            if escaped not in JSONTokenizer._ESCAPES:
                raise ValueError(f"Invalid escape sequence '\\{escaped}'")
            if escaped == "u":
                digits = input[escape + 2 : escape + 6]
                if len(digits) < 4 or any(
                    d not in "0123456789abcdefABCDEF" for d in digits
                ):
                    raise ValueError(f"Invalid unicode escape '{digits}'")
                current = escape + 6
                continue
            # Skip both characters, so `\"` reads as content, not as the end.
            current = escape + 2

    @staticmethod
    def tokenize(input: str) -> list[JSONToken]:
        """Split `input` into tokens, dropping the whitespace.

        A trailing word that could still grow into a valid token (a
        number fragment, a keyword prefix, a string with no closing
        quote) comes out as `Incomplete`, so a caller can tell "not a
        token yet" from "never a token".
        """
        tokens: list[JSONToken] = []
        current = 0
        length = len(input)
        while current < length:
            char = input[current]
            punctuation = JSONTokenizer._PUNCTUATION.get(char)
            if punctuation is not None:
                tokens.append(JSONToken(punctuation, char))
                current += 1
            elif char == '"':
                end = JSONTokenizer.__find_closing_quote(input, current + 1)
                if end == -1:
                    # No closing quote: the rest is an unfinished string.
                    tokens.append(
                        JSONToken(
                            JSONTokenType.Incomplete, input[current + 1 :]
                        )
                    )
                    current = length
                else:
                    tokens.append(
                        JSONToken(
                            JSONTokenType.String, input[current + 1 : end]
                        )
                    )
                    current = end + 1
            elif char.isalnum() or char in "_-":
                value, current = JSONTokenizer.__scan_word(input, current)
                tokens.append(JSONTokenizer.__tokenize_word(value))
            elif char.isspace():
                current += 1
            else:
                raise ValueError(f"Unexpected character '{char}'")
        return tokens
