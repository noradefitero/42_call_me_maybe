"""A countdown that lets you skip the wait with any key."""

import select
import sys
import termios
import time
import tty
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import IO

from rich.console import Group
from rich.text import Text

COUNTDOWN_SECONDS = 8
"""How long the program waits before closing."""

TICK = 1.0
"""How often the countdown is checked for a key."""

COPY = "Closing in {seconds}s… press any key to skip"
CLOSING = "Closing…"
"""Once the seconds are up there is nothing left to skip."""


def countdown(
    group: Group,
    seconds: int = COUNTDOWN_SECONDS,
) -> None:
    """Count `seconds` down in `group`, until a key or the end.

    The line repaints itself, so whoever holds `group` only has to
    keep refreshing the screen, as the `Live` display already does.
    Nothing is drawn and nothing is waited for when there is no
    terminal to read keys from, so a piped run is never slowed down by
    a wait nobody can interrupt.
    """
    if not _interactive():
        return
    with _raw_keys(sys.stdin):
        _tick_down(group, seconds)


@contextmanager
def _raw_keys(stdin: IO[str]) -> Iterator[None]:
    """Hand every keypress to the caller, one at a time and untyped.

    A terminal that refuses the change is left exactly as it was: the
    countdown then can't be skipped, and the shell is never left
    halfway back to line editing.
    """
    try:
        fd = stdin.fileno()
        settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
    except (termios.error, OSError, ValueError):
        # No terminal to talk to. Read whatever turns up and move on.
        yield
        return
    try:
        yield
    finally:
        with suppress(termios.error, OSError):
            termios.tcsetattr(fd, termios.TCSADRAIN, settings)


def _interactive() -> bool:
    """Whether there is a terminal on stdin to read keys from."""
    try:
        return bool(sys.stdin.isatty())
    except (AttributeError, ValueError):
        # A closed stdin has nobody left to press a key
        return False


def _tick_down(group: Group, seconds: int) -> None:
    """Put the countdown in `group` as its newest entry and run it."""
    line = Text(_line(seconds), style="dim")
    group.renderables.append(line)
    deadline = time.monotonic() + seconds
    left = seconds
    while left > 0 and not _key_pressed(deadline):
        left -= 1
        line.plain = _line(left)


def _line(left: int) -> str:
    """The countdown line: seconds left, or a plain goodbye."""
    if left <= 0:
        return CLOSING
    return COPY.format(seconds=left)


def _key_pressed(deadline: float) -> bool:
    """Wait a second for a key, and say whether one turned up.

    The wait shrinks as the deadline nears, and an input that is done
    reading counts as a key: there is nobody left to press one.
    """
    timeout = min(TICK, deadline - time.monotonic())
    if timeout <= 0:
        return False
    try:
        readable, _, _ = select.select([sys.stdin], [], [], timeout)
    except (OSError, ValueError):
        return True
    if not readable:
        return False
    try:
        sys.stdin.read(1)
    except (OSError, UnicodeDecodeError):
        pass
    return True
