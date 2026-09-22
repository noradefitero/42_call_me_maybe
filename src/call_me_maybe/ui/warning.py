from rich.text import Text


class Warning(Text):
    def __init__(self, text: str = "") -> None:
        super().__init__(f"Warning: {text}\n", style="dark_orange bold")
