import asyncio

import numpy as np
from llm_sdk import Small_LLM_Model
from rich import print
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from call_me_maybe.config import DEFAULT_MODEL
from call_me_maybe.prompt import Prompt
from call_me_maybe.ui.error import Error
from call_me_maybe.ui.spinner import loading_indicator
from call_me_maybe.ui.warning import Warning


class Generator:
    """Streams an LLM reply into the terminal with a typing animation."""

    MAX_TOKENS = 100
    REFRESH_PER_SECOND = 25

    USER_PROMPT_DELAY = 0.02
    ASSISTANT_DELAY = 0.02

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        print(
            Text(
                f"\nOpening model {model}...",
                style="cyan",
            )
        )
        try:
            self.llm = Small_LLM_Model(model)
        except OSError as e:
            print(
                Error(e),
                Warning("Running default model as fallback"),
                Text(f"\nOpening model {DEFAULT_MODEL}...", style="cyan"),
            )
            self.llm = Small_LLM_Model(DEFAULT_MODEL)
        print()

    async def run_prompt(self, prompt: Prompt) -> str:
        """Animate the prompt and its answer, then return the answer."""
        user_panel = Panel(
            "",
            title="User Prompt",
            title_align="left",
            expand=False,
            style="red",
        )
        assistant_panel = Panel(
            "",
            title="Assistant",
            title_align="left",
            expand=False,
            style="yellow",
        )
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        with Live(
            user_panel,
            refresh_per_second=self.REFRESH_PER_SECOND,
        ) as live:
            generate_task = asyncio.create_task(
                self.__generate_tokens(prompt, queue)
            )
            async with loading_indicator(user_panel, "User Prompt"):
                await self.__animate(
                    f">> {prompt.user_prompt}",
                    user_panel,
                    delay=self.USER_PROMPT_DELAY,
                )
            live.update(Group(user_panel, assistant_panel))
            output = await self.__stream_tokens(
                queue,
                assistant_panel,
                title="Assistant",
                delay=self.ASSISTANT_DELAY,
            )
            await generate_task
            assistant_panel.style = "green"
            live.update(Group(user_panel, assistant_panel))
        return output

    async def __stream_tokens(
        self,
        queue: asyncio.Queue[str | None],
        panel: Panel,
        *,
        title: str,
        delay: float,
    ) -> str:
        """Animate every token from `queue` into `panel` and join them."""
        tokens: list[str] = []
        async with loading_indicator(panel, title):
            while True:
                token = await queue.get()
                if token is None:
                    break
                tokens.append(token)
                await self.__animate(token, panel, delay=delay)
        return "".join(tokens)

    async def __animate(
        self,
        text: str,
        panel: Panel,
        *,
        delay: float = 0.0,
    ) -> None:
        """Type `text` into `panel` one character at a time."""
        for char in text:
            panel.renderable = str(panel.renderable) + char
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
