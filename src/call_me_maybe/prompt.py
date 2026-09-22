from string import Template

import toon_format

from call_me_maybe.models.function_lists import FunctionDefinitionList


class Prompt:
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
        definitions_json = (
            definitions.model_dump_json() if definitions is not None else "{}"
        )
        self.__definitions = toon_format.encode(definitions_json)
        self.__user_prompt = user_prompt

    @property
    def user_prompt(self) -> str:
        return self.__user_prompt

    @user_prompt.setter
    def user_prompt(self, msg: str) -> None:
        self.__user_prompt = msg

    def __str__(self) -> str:
        return self.__template.safe_substitute(
            system=self.__system,
            definitions=self.__definitions,
            prompt=self.__user_prompt,
        )
