from pathlib import Path
from typing import Annotated

import typer
from rich import box, print
from rich.console import Group
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
from call_me_maybe.pipeline.parse import run as parse_run
from call_me_maybe.pipeline.runner import run as runner_run
from call_me_maybe.ui.error import Error
from call_me_maybe.ui.rainbow import RainbowHighlighter

app = typer.Typer()


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
    rainbow = RainbowHighlighter()
    print(
        Panel(
            Group(
                rainbow(
                    Text(
                        "************* CALL ME MAYBE *************",
                        style="bold",
                        justify="center",
                    ),
                ),
                Text(
                    "by Nora de Fitero Teijeira (@noradefitero)",
                    style="bold",
                    justify="center",
                ),
            ),
            box=box.DOUBLE,
            expand=False,
            padding=(1, 8),
            style="cyan",
        ),
        "\n",
    )
    try:
        args = parse_run(
            system_prompt=system_prompt,
            system_prompt_file=system_prompt_file,
            functions_definition=functions_definition,
            input_file=input_file,
        )
    except PromptLoadError as e:
        print(Error(e))
        raise typer.Exit(code=1)
    runner_run(args, model)
