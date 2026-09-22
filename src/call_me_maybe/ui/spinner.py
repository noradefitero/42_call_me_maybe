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
