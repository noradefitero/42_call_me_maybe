from typing import Any

from pydantic import BaseModel


class FunctionCall(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]
