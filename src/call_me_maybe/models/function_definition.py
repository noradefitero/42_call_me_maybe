from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class JsonType(TypedDict):
    """The JSON Schema of a value, as its single `type` key."""

    type: str


class FunctionDefinition(BaseModel):
    """A function the model is offered, and what it takes and gives."""

    name: str = Field(description="Name the model uses to call it")
    description: str = Field(description="What the function does")
    parameters: dict[str, JsonType] = Field(
        description="Each argument, keyed by name, with its type"
    )
    returns: JsonType = Field(description="Type the function returns")
