from pathlib import Path
from time import sleep
from typing import cast

from rich.console import Group
from rich.layout import Layout
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from call_me_maybe.config import UI_LOADING_DELAY
from call_me_maybe.exceptions import OutputWriteError
from call_me_maybe.models.function_call import FunctionCall
from call_me_maybe.models.function_lists import FunctionCallList


def __step(main_group: Group, running: str, done: str) -> None:
    """Add a panel to `main_group` that shows `running` until it is done.

    The spinner inside the panel turns into `done` in green, so the
    finished steps stay on screen in the order they happened.
    """
    group = Group()
    main_group.renderables.append(Panel(group))
    group.renderables.append(Spinner("dots", text=running))
    sleep(UI_LOADING_DELAY)
    group.renderables.clear()
    group.renderables.append(Text(done, style="green"))


def run(
    function_calls: list[FunctionCall], output_file: Path, layout: Layout
) -> None:
    """Write the collected calls to `output_file` as JSON.

    Raises OutputWriteError if the directory or the file cannot be
    written, so the caller can report it instead of crashing.
    """
    main_panel = cast(Panel, layout["main"].renderable)
    main_group = cast(Group, main_panel.renderable)
    layout["sidebar"].visible = False
    main_panel.title = None
    __step(
        main_group,
        "Dumping processed data to raw JSON",
        "Dumped processed data to raw JSON",
    )
    to_write = FunctionCallList(function_calls).model_dump_json(indent=4)
    __step(main_group, "Writing data to output", "Wrote data to output")
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(to_write)
    except OSError as e:
        __step(main_group, "Writing data to output", "")
        main_group.renderables[-1] = Text(
            f"Failed writing data to {output_file}: {e}", style="red"
        )
        raise OutputWriteError(
            f"Failed writing results to '{output_file}': {e}"
        ) from e
