from typing import Any

from rich import print as rich_print
from rich.console import RenderableType


def print(
    *objects: Any,
    output: RenderableType | None = None,
) -> None:
    if output is not None:
        output.renderables.extend(objects)
    else:
        rich_print(objects)
