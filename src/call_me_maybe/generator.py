import asyncio

import numpy as np
from llm_sdk import Small_LLM_Model
from rich.console import Group
from rich.panel import Panel
from rich.spinner import Spinner
from rich.style import StyleType
from rich.text import Text

from call_me_maybe.config import DEFAULT_MODEL
from call_me_maybe.logger import logger
from call_me_maybe.prompt import Prompt
from call_me_maybe.ui.spinner import SpinnerText


class ModelNotLoaded(Exception):
    pass


class Generator:
    """Streams an LLM reply into the terminal with a typing animation."""

    MAX_TOKENS = 100
    REFRESH_PER_SECOND = 25

    USER_PROMPT_DELAY = 0.02
    ASSISTANT_DELAY = 0.02

    USER_PANEL_TITLE = "User Panel"
    ASSISTANT_PANEL_TITLE = "Assistant"

    def __init__(
        self, model: str = DEFAULT_MODEL, output: Group | None = None
    ) -> None:
        self.__output = output
        try:
            self.__init_model(model)
        except ModelNotLoaded as e:
            logger.error(e)
            logger.warning(
                f"Running default model as fallback: {DEFAULT_MODEL}"
            )
            self.__init_model(DEFAULT_MODEL)

    def __init_model(self, model: str) -> None:
        if self.__output:
            text = Text(f"Opening model {model}", style="cyan")
            spinner = Spinner("dots", text=text, style="cyan")
            self.__output.renderables.append(spinner)
        try:
            self.llm = Small_LLM_Model(model)
        except OSError as e:
            if self.__output:
                text = Text(f"X {text}", style="red")
            raise ModelNotLoaded(e)
        finally:
            self.__output.renderables.remove(spinner)
            self.__output.renderables.append(text)

    async def run_prompt(self, prompt: Prompt, group: Group) -> str:
        """Animate the prompt and its answer, then return the answer."""
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        generate_task = asyncio.create_task(
            self.__generate_tokens(prompt, queue)
        )
        user_panel = Panel(
            "",
            title=self.USER_PANEL_TITLE,
            title_align="left",
            expand=False,
            style="red",
        )
        assistant_panel = Panel(
            "",
            title=self.ASSISTANT_PANEL_TITLE,
            title_align="left",
            expand=False,
            style="yellow",
        )
        group.renderables.append(user_panel)
        user_panel.title = SpinnerText("dots", text=self.USER_PANEL_TITLE)
        await self.__animate(
            f">> {prompt.user_prompt}",
            user_panel,
            delay=self.USER_PROMPT_DELAY,
            style="default",
        )
        user_panel.title = self.USER_PANEL_TITLE
        group.renderables.append(assistant_panel)
        assistant_panel.title = SpinnerText(
            "dots", text=self.ASSISTANT_PANEL_TITLE
        )

        output = await self.__stream_tokens(
            queue, assistant_panel, delay=self.ASSISTANT_DELAY
        )
        await generate_task
        assistant_panel.title = self.ASSISTANT_PANEL_TITLE
        assistant_panel.style = "green"
        return output

    async def __stream_tokens(
        self,
        queue: asyncio.Queue[str | None],
        panel: Panel,
        *,
        delay: float,
    ) -> str:
        """Animate every token from `queue` into `panel` and join them."""
        tokens: list[str] = []
        while True:
            token = await queue.get()
            if token is None:
                break
            tokens.append(token)
            await self.__animate(token, panel, delay=delay, style="default")
        return "".join(tokens)

    async def __animate(
        self,
        text: str,
        panel: Panel,
        style: StyleType | None = None,
        *,
        delay: float = 0.0,
    ) -> None:
        """Type `text` into `panel` one character at a time."""
        for char in text:
            panel.renderable = Text(f"{panel.renderable}{char}", style=style)
            if delay:
                await asyncio.sleep(delay)

    async def __generate_tokens(
        self,
        prompt: Prompt,
        queue: asyncio.Queue[str | None],
    ) -> None:
        """Generate tokens and put each one on `queue`, ending with None."""
        input_ids = self.llm.encode(str(prompt))[0].tolist()
        for _ in range(self.MAX_TOKENS):
            logits = await asyncio.to_thread(
                self.llm.get_logits_from_input_ids,
                input_ids,
            )
            token_id = int(np.argmax(logits))
            input_ids.append(token_id)
            await queue.put(self.llm.decode(token_id))
        await queue.put(None)
