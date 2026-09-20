from typing import Any

from pydantic import BaseModel


class FnCalling(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]
