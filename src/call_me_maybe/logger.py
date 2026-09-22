import logging
from threading import Lock

from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.segment import Segment
from rich.text import Text

_LOG_STYLE_BY_LEVEL: dict[int, str] = {
    logging.DEBUG: "dim",
    logging.INFO: "",
    logging.WARNING: "dark_orange bold",
    logging.ERROR: "red bold",
    logging.CRITICAL: "bold red reverse",
}


class RichGroupHandler(logging.Handler):
    def __init__(self, max_lines: int = 100) -> None:
        super().__init__()

        self.max_lines = max_lines
        self.messages: list[Text] = []
        self._lock = Lock()

    def emit(self, record: logging.LogRecord) -> None:
        message = self.format(record)
        style = _LOG_STYLE_BY_LEVEL.get(record.levelno, "")

        with self._lock:
            self.messages.append(Text(message, style=style))

            if len(self.messages) > self.max_lines:
                del self.messages[: -self.max_lines]

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        with self._lock:
            messages = list(self.messages)

        if options.height is None or options.height <= 0:
            yield Group(*messages)
            return

        # Follow the tail by *rendered lines*, not messages: multiline or
        # wrapped entries would otherwise overflow the region and be cropped
        # from the bottom, hiding the newest logs.
        lines = console.render_lines(
            Group(*messages), options.update_height(None)
        )
        new_line = Segment.line()
        for line in lines[-options.height :]:
            yield from line
            yield new_line


logger = logging.getLogger("call_me_maybe")
logger.setLevel(logging.INFO)

handler = RichGroupHandler(max_lines=100)

handler.setFormatter(
    logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
)

logger.addHandler(handler)
