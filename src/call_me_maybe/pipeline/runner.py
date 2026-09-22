import asyncio

from rich.console import Group
from rich.layout import Layout
from rich.panel import Panel

from call_me_maybe.generator import Generator
from call_me_maybe.pipeline.parse import ParsedArgs
from call_me_maybe.prompt import Prompt


def run(args: ParsedArgs, model: str, layout: Layout) -> list[str]:
    """Run the function-calling pipeline, printing everything into `layout`."""
    response: list[str] = []
    prompt = Prompt(
        template=args["template"],
        system=args["system_prompt"],
        definitions=args["definitions"],
    )
    chat_group: Panel = layout["chat"].renderable.renderable
    sidebar_group: Group = layout["sidebar"].renderable.renderable
    generator = Generator(model, output=chat_group)
    layout["sidebar"].visible = True
    layout["chat"].renderable.title = "Chat"
    for msg in args["input"]:
        prompt.user_prompt = msg
        new_group = Group()
        chat_group.renderables.append(new_group)
        r = asyncio.run(generator.run_prompt(prompt, new_group))
        response.append(r)
        sidebar_group.renderables.append(
            Panel(r, title=f"Response {len(response)}", title_align="left")
        )
    return response
