from string import Template

import toon_format

from call_me_maybe.models.function_lists import FunctionDefinitionList


class Prompt:
    """A template with the system prompt, definitions and user message."""

    def __init__(
        self,
        *,
        template: str,
        system: str = "",
        definitions: FunctionDefinitionList | None = None,
        user_prompt: str = "",
    ) -> None:
        self.__template = Template(template)
        self.__system = system
        self.__function_definitions = definitions or FunctionDefinitionList([])
        # The grammar needs a real list even when none was given, but
        # the model is shown "{}" rather than an empty list
        self.__definitions = toon_format.encode(
            definitions.model_dump_json() if definitions is not None else "{}"
        )
        self.__user_prompt = user_prompt

    @property
    def function_definitions(self) -> FunctionDefinitionList:
        """The functions on offer, which the grammar walks."""
        return self.__function_definitions

    @property
    def user_prompt(self) -> str:
        """The user message being answered."""
        return self.__user_prompt

    @user_prompt.setter
    def user_prompt(self, msg: str) -> None:
        """Replace the user message before the next answer."""
        self.__user_prompt = msg

    def __str__(self) -> str:
        return self.__template.safe_substitute(
            system=self.__system,
            definitions=self.__definitions,
            prompt=self.__user_prompt,
        )
