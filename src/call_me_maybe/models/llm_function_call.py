from typing import Any

from pydantic import BaseModel, Field


class LLMFunctionCall(BaseModel):
    """The call the model answers with, before the input is added."""

    name: str = Field(description="Name of the function called")
    parameters: dict[str, Any] = Field(
        description="Arguments the function was called with"
    )
