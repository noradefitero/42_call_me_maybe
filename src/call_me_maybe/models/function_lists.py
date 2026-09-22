from pydantic import RootModel

from .function_calling import FunctionCall
from .function_definition import FunctionDefinition

FunctionDefinitionList = RootModel[list[FunctionDefinition]]
FunctionCallList = RootModel[list[FunctionCall]]
