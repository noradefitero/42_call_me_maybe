import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from rich import print

from call_me_maybe.config import (
    PROMPT_TEMPLATE_PATH,
    SYSTEM_PROMPT_DEFAULT_PATH,
)
from call_me_maybe.exceptions import PromptLoadError
from call_me_maybe.models import FunctionDefinitionList
from call_me_maybe.models.input_list import InputList
from call_me_maybe.ui.warning import Warning


def get_prompt_template() -> str:
    try:
        return PROMPT_TEMPLATE_PATH.read_text()
    except (
        FileNotFoundError,
        PermissionError,
        UnicodeDecodeError,
        OSError,
    ) as e:
        raise PromptLoadError(f"Failed reading prompt template: {e}")


def get_system_prompt_from_file(
    system_prompt_file: Path | None = None,
) -> str:
    """Read a system prompt file, falling back to the bundled default."""
    if system_prompt_file:
        try:
            print(f"Loading custom system prompt at: {system_prompt_file}")
            return system_prompt_file.read_text()
        except (
            FileNotFoundError,
            PermissionError,
            UnicodeDecodeError,
            OSError,
        ) as e:
            print(
                Warning(
                    "Failed opening user defined system prompt file,"
                    f"using default system prompt: {e}"
                )
            )
    try:
        return SYSTEM_PROMPT_DEFAULT_PATH.read_text()
    except (
        FileNotFoundError,
        PermissionError,
        UnicodeDecodeError,
        OSError,
    ) as e:
        raise PromptLoadError(f"Failed reading system prompt: {e}")


def get_function_definitions(path: Path) -> FunctionDefinitionList:
    """Load and validate a JSON file describing available functions."""
    try:
        print(f"Loading fuction definitions at: {path}")
        return FunctionDefinitionList.model_validate_json(path.read_text())
    except (
        FileNotFoundError,
        PermissionError,
        UnicodeDecodeError,
        OSError,
    ) as e:
        raise PromptLoadError(f"Failed reading prompt template: {e}")
    except ValidationError as e:
        raise PromptLoadError(f"Incorrect format in prompt template: {e}")


def get_prompt_hook(dct: dict[Any, Any]) -> Any:
    if "prompt" not in dct:
        raise PromptLoadError(
            f"Prompt value is not defined in input message {dct}"
        )
    if len(dct) != 1:
        raise PromptLoadError(f"Unknown values in input message {dct}")
    return dct["prompt"]


def get_input_data(path: Path) -> list[str]:
    try:
        print(f"Loading input data at: {path}")
        text = path.read_text()
    except (
        FileNotFoundError,
        PermissionError,
        UnicodeDecodeError,
        OSError,
    ) as e:
        raise PromptLoadError(f"Failed reading input messages: {e}")
    try:
        parsed: list[str] = json.loads(text, object_hook=get_prompt_hook)
    except json.JSONDecodeError as e:
        raise PromptLoadError(f"Invalid JSON format in input messages: {e}")
    try:
        InputList(parsed)
    except ValidationError as e:
        raise PromptLoadError(f"Incorrect format in input messages: {e}")
    return parsed
