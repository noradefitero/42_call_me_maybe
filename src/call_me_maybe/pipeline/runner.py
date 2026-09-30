import asyncio
from time import sleep
from typing import Any, cast

from pydantic import ValidationError
from rich.console import Group, RenderableType
from rich.layout import Layout
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from call_me_maybe.config import UI_LOADING_DELAY
from call_me_maybe.exceptions import NoValidCallError
from call_me_maybe.generator import Generator
from call_me_maybe.logger import logger
from call_me_maybe.models.function_call import FunctionCall
from call_me_maybe.models.function_lists import FunctionDefinitionList
from call_me_maybe.models.llm_function_call import LLMFunctionCall
from call_me_maybe.pipeline.parse import ParsedArgs
from call_me_maybe.prompt import Prompt

MAX_RETRIES = 5


def __show_status(run_info: Group, status: RenderableType) -> None:
    """Show `status` on its own in `run_info`, dropping what was there."""
    run_info.renderables.clear()
    run_info.renderables.append(status)


def __show_done(run_info: Group, spinner: Spinner, message: str) -> None:
    """Swap the `spinner` in `run_info` for `message` in green."""
    run_info.renderables.remove(spinner)
    run_info.renderables.append(Text(message, style="green"))


def __input_panel(sidebar_group: Group, index: int, msg: str) -> Group:
    """Add the sidebar entry for input `index` to `sidebar_group`.

    Returns the group inside it, where the progress of that input goes.
    """
    run_info = Group()
    sidebar_group.renderables.append(
        Panel(
            Group(
                Text(f'Input Nº{index} "{msg}":\n'),
                run_info,
            )
        )
    )
    return run_info


def __answer(
    generator: Generator,
    prompt: Prompt,
    new_group: Group,
    run_info: Group,
    msg: str,
    definitions: FunctionDefinitionList,
) -> FunctionCall:
    """Ask the model to answer `msg` and validate what comes back.

    Every attempt is reported in `run_info`, and the assistant panel
    turns green once an answer is accepted and red if it is not.

    Returns:
        The validated call.

    Raises:
        NoValidCallError: if no attempt produced a schema-valid call.
            Nothing is invented to fill the gap, because a placeholder
            would not match the schema.
    """
    answering_spinner = Spinner("dots", text="Answering to input")
    for attempt in range(MAX_RETRIES):
        __show_status(run_info, answering_spinner)
        answer, assistant_panel = asyncio.run(
            generator.run_prompt(prompt, new_group, attempt)
        )
        if answer is None:
            # The panel is already red; the next attempt picks a different
            # token, so it can get out of the deadlock.
            logger.warning(f"Grammar deadlock, retry {attempt}/{MAX_RETRIES}")
            answering_spinner = Spinner(
                "dots",
                text=f"Answering to input. Attempt {attempt}/{MAX_RETRIES}",
                style="dark orange",
            )
            __show_status(
                run_info, Text("Failed answering to input", style="red")
            )
            continue
        __show_done(run_info, answering_spinner, "LLM answered to input")

        validating_spinner = Spinner(
            "dots", text="Validating answer with Pydantic"
        )
        run_info.renderables.append(validating_spinner)
        sleep(UI_LOADING_DELAY)
        try:
            call = LLMFunctionCall.model_validate_json(answer)
        except ValidationError:
            logger.warning(
                f"Failed validating LLM answer, retry {attempt}/{MAX_RETRIES}"
            )
            answering_spinner = Spinner(
                "dots",
                text=f"Answering to input. Attempt {attempt}/{MAX_RETRIES}",
                style="dark orange",
            )
            assistant_panel.style = "red"
            __show_status(
                run_info, Text("Failed answering to input", style="red")
            )
            continue

        __show_done(run_info, validating_spinner, "Validated answer format")
        assistant_panel.style = "green"
        return FunctionCall(
            prompt=msg,
            name=call.name,
            parameters=__typed_parameters(call, definitions),
        )
    logger.error(f"No valid call produced for prompt: {msg}")
    raise NoValidCallError(
        f"No schema-valid call for the prompt {msg!r} after "
        f"{MAX_RETRIES} attempts. No output file was written: filling "
        f"the gap with a placeholder would not match the schema."
    )


def __typed_parameters(
    call: LLMFunctionCall, definitions: FunctionDefinitionList
) -> dict[str, Any]:
    """Give each argument the type the chosen function declares.

    A numeric argument is written as a float, the form the subject
    shows in its example output. Every other argument is left as the
    model produced it, since the grammar already fixed its type.
    """
    definition = next(
        (f for f in definitions.root if f.name == call.name), None
    )
    if definition is None:
        return call.parameters
    typed: dict[str, Any] = {}
    for key, value in call.parameters.items():
        spec = definition.parameters.get(key)
        declared = spec.get("type") if spec is not None else None
        # `bool` is a subclass of `int`, so it is not a number here.
        is_int = isinstance(value, int) and not isinstance(value, bool)
        if declared in ("number", "float") and is_int:
            typed[key] = float(value)
        else:
            typed[key] = value
    return typed


def run(args: ParsedArgs, model: str, layout: Layout) -> list[FunctionCall]:
    """Run the function-calling pipeline, printing everything into `layout`."""
    prompt = Prompt(
        template=args["template"],
        system=args["system_prompt"],
        definitions=args["definitions"],
    )
    main_panel = cast(Panel, layout["main"].renderable)
    sidebar_panel = cast(Panel, layout["sidebar"].renderable)
    main_group = cast(Group, main_panel.renderable)
    sidebar_group = cast(Group, sidebar_panel.renderable)
    generator = Generator(model, output=main_group)
    layout["sidebar"].visible = True
    main_panel.title = "Chat"
    response: list[FunctionCall] = []
    definitions = args["definitions"]
    for idx, msg in enumerate(args["input"]):
        prompt.user_prompt = msg
        # Its own group, so the answer and its retries stay together.
        new_group = Group()
        main_group.renderables.append(new_group)
        run_info = __input_panel(sidebar_group, idx, msg)
        response.append(
            __answer(generator, prompt, new_group, run_info, msg, definitions)
        )
    return response
