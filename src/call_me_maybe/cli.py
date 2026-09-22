from pathlib import Path
from typing import Annotated

import typer
from rich import print
from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from call_me_maybe import __version__
from call_me_maybe.config import (
    DEFAULT_FUNCTIONS_FILE,
    DEFAULT_INPUT_FILE,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_FILE,
)
from call_me_maybe.exceptions import PromptLoadError
from call_me_maybe.logger import handler
from call_me_maybe.pipeline.parse import run as parse_run
from call_me_maybe.pipeline.runner import run as runner_run
from call_me_maybe.ui.header import Header

app = typer.Typer()


def make_layouts() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", ratio=3),
        Layout(name="console", ratio=1),
    )
    layout["main"].split_row(
        Layout(Group(), name="chat", ratio=3),
        Layout(Group(), name="sidebar", ratio=2, visible=False),
    )
    return layout


@app.command()
def run(
    functions_definition: Annotated[
        Path, typer.Option("--functions-definition", "-f")
    ] = DEFAULT_FUNCTIONS_FILE,
    input_file: Annotated[
        Path, typer.Option("--input", "-i")
    ] = DEFAULT_INPUT_FILE,
    output_file: Annotated[
        Path, typer.Option("--output", "-o")
    ] = DEFAULT_OUTPUT_FILE,
    system_prompt: Annotated[
        str | None, typer.Option("--system-prompt")
    ] = None,
    system_prompt_file: Annotated[
        Path | None, typer.Option("--system-prompt-file")
    ] = None,
    model: Annotated[str, typer.Option("--model", "-m")] = DEFAULT_MODEL,
    version: Annotated[bool, typer.Option("--version")] = False,
) -> None:
    if version:
        print(f"call-me-maybe {__version__}")
        raise typer.Exit()
    layout = make_layouts()
    layout["header"].update(Header())
    layout["console"].update(Panel(handler, title="Logs"))
    layout["main"]["chat"].update(Panel(Group()))
    layout["main"]["sidebar"].update(Panel(Group(), title="Sidebar"))
    with Live(layout, refresh_per_second=20, screen=True) as live:
        try:
            args = parse_run(
                system_prompt=system_prompt,
                system_prompt_file=system_prompt_file,
                functions_definition=functions_definition,
                input_file=input_file,
            )
        except PromptLoadError as e:
            # Leave the Live first so the error stays visible on the terminal.
            live.stop()
            print(Text(e, style="red bold"))
            raise typer.Exit(code=1)
        runner_run(args, model, layout["main"])
