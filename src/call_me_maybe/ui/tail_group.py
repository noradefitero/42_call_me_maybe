"""A `Group` that scrolls to the newest entries of a panel."""

from rich.console import (
    Console,
    ConsoleOptions,
    Group,
    RenderableType,
    RenderResult,
)
from rich.segment import Segment
from rich.text import Text


class TailGroup(Group):
    """A `Group` that keeps the newest entries in view.

    The chat and the sidebar grow with every input, and a region that
    overflows is cropped from the top, which buries whatever was just
    written. This one measures its entries from the end and only shows
    the ones the window still holds, plus a discreet line counting the
    ones left out.

    The measurement is the render, and `Live` renders the layout on
    every refresh, so spinners inside the visible entries keep
    animating.
    """

    def __init__(
        self, *renderables: RenderableType, entry: str = "message"
    ) -> None:
        super().__init__(*renderables)
        self.entry = entry

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        window: int | None = (
            options.height
            if options.height is not None
            else options.max_height
        )
        if window is None or window <= 0:
            # Nothing bounds the render, so there is nothing to scroll.
            yield from self.renderables
            return
        # Every entry is measured at its own height: keeping the
        # window's limit would crop the last one for us, and which
        # entries to cut belongs here.
        measured = options.update(height=None)
        visible: list[list[Segment]] = []
        hidden = 0
        for index in range(len(self.renderables) - 1, -1, -1):
            lines = console.render_lines(self.renderables[index], measured)
            if len(visible) + len(lines) <= window:
                visible[:0] = lines
                continue
            if not visible:
                # The newest entry is taller than the window. Its
                # bottom is what matters: that's the answer being
                # typed right now. Only the entries below it are cut.
                hidden = index
            else:
                # Entries 0..index are cut; index+1..end are shown.
                hidden = index + 1
            break
        if hidden:
            # The notice takes the row the entries left unused, or the
            # top line of the oldest one when the window is exactly
            # full: that line reads as the entry going on above the
            # viewport.
            if len(visible) >= window:
                del visible[0]
            notice = console.render_lines(self.__notice(hidden), measured)
            for line in notice:
                yield from line
                yield Segment.line()
        new_line = Segment.line()
        for line in visible:
            yield from line
            yield new_line

    def __notice(self, hidden: int) -> Text:
        """A dim line counting the entries left out of the window."""
        entry = self.entry if hidden == 1 else f"{self.entry}s"
        return Text(
            f"… {hidden} earlier {entry} hidden",
            style="dim",
            no_wrap=True,
            overflow="ellipsis",
        )
