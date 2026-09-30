from typing import Any

from pydantic import BaseModel


class LLMFunctionCall(BaseModel):
    name: str
    parameters: dict[str, Any]
