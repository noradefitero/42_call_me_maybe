from rich.columns import Columns
from rich.panel import Panel
from rich.text import Text

from call_me_maybe.ui.rainbow import RainbowHighlighter


class Header:
    """Display header with clock."""

    def __rich__(self) -> Panel:
        rainbow = RainbowHighlighter()
        columns = Columns(
            [
                rainbow(
                    Text(
                        "Welcome to CALL ME MAYBE",
                        style="bold",
                    ),
                ),
                Text(
                    "by Nora de Fitero Teijeira (@noradefitero)",
                ),
            ]
        )
        return Panel(columns)
