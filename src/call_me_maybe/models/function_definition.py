from pydantic import BaseModel
from typing_extensions import TypedDict


class JsonType(TypedDict):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, JsonType]
    returns: JsonType
