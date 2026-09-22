from pathlib import Path

from typing_extensions import TypedDict

from call_me_maybe.io.read import (
    get_function_definitions,
    get_input_data,
    get_prompt_template,
    get_system_prompt_from_file,
)
from call_me_maybe.models.function_lists import FunctionDefinitionList


class ParsedArgs(TypedDict):
    template: str
    system_prompt: str
    definitions: FunctionDefinitionList
    input: list[str]


def run(
    system_prompt: str | None,
    system_prompt_file: Path | None,
    functions_definition: Path,
    input_file: Path,
) -> ParsedArgs:
    if system_prompt is None:
        system_prompt = get_system_prompt_from_file(system_prompt_file)
    template = get_prompt_template()
    definitions = get_function_definitions(functions_definition)
    input = get_input_data(input_file)
    return {
        "template": template,
        "system_prompt": system_prompt,
        "definitions": definitions,
        "input": input,
    }
