from typing import Any

from pydantic import BaseModel, Field


class FunctionCall(BaseModel):
    """A validated function call, as it goes to the results file."""

    prompt: str = Field(description="Input message it answers")
    name: str = Field(description="Name of the function called")
    parameters: dict[str, Any] = Field(
        description="Arguments the function was called with"
    )
