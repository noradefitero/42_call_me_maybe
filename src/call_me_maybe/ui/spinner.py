"""A `Spinner` that can be used as a `Panel.title`."""

from time import monotonic
from typing import cast

from rich.spinner import Spinner
from rich.text import Text


class SpinnerText(Spinner):
    """A `Spinner` that can be used as a `Panel.title`.

    `Panel` renders a non-string title by calling `copy()`, which
    `Spinner` doesn't implement. This wrapper returns the current
    animation frame as a `Text` instead.
    """

    def copy(self) -> Text:
        return cast(Text, self.render(monotonic()))

    def __copy__(self) -> Text:
        return self.copy()


def spinner_title(text: str) -> Text:
    """An animating spinner to use as a `Panel.title`.

    `Panel.title` is typed as `str | Text | None` although it renders
    any `RichCast`, so the cast tells mypy what rich does at runtime.
    """
    return cast(Text, SpinnerText("dots", text=text))
