"""Reusable async loading indicator for Rich panels."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from itertools import cycle

from rich.panel import Panel

DEFAULT_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
DEFAULT_DELAY = 0.1


@asynccontextmanager
async def loading_indicator(
    panel: Panel,
    title: str,
    *,
    frames: str = DEFAULT_FRAMES,
    delay: float = DEFAULT_DELAY,
) -> AsyncIterator[None]:
    """Show a spinner in `panel.title` while the body runs.

    The spinner cycles through `frames` every `delay` seconds. The
    original `title` is restored once the body finishes.
    """
    task = asyncio.create_task(_spin(panel, title, frames, delay))
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    panel.title = title


async def _spin(
    panel: Panel,
    title: str,
    frames: str,
    delay: float,
) -> None:
    """Cycle spinner frames in `panel.title` until cancelled."""
    for frame in cycle(frames):
        panel.title = f"{title} {frame}"
        await asyncio.sleep(delay)
