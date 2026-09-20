from pydantic import RootModel

from .fn_definition import FnDefinition

FnList = RootModel[list[FnDefinition]]
