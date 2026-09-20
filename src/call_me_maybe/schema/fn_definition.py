from pydantic import BaseModel
from typing_extensions import TypedDict


class FnType(TypedDict):
    type: str


class FnDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, FnType]
    returns: FnType
