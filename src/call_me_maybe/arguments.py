from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Arguments:
    functions_definition: Path
    input: Path
    output: Path
    system_prompt: str
    model: str
