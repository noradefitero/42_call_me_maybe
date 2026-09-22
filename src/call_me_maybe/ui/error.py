from typing import Any

from rich.panel import Panel


class Error(Panel):
    def __init__(self, err: Any) -> None:
        super().__init__(
            str(err),
            style="red bold",
            title=f"Error {type(err).__name__}",
            title_align="left",
            padding=(1, 2),
        )
