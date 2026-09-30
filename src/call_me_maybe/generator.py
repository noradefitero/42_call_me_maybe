import asyncio

import numpy as np
from rich.console import Group
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from call_me_maybe.config import DEFAULT_MODEL
from call_me_maybe.exceptions import ModelNotLoaded
from call_me_maybe.json_grammar import JSONGrammar
from call_me_maybe.logger import logger
from call_me_maybe.models.function_lists import FunctionDefinitionList
from call_me_maybe.prompt import Prompt
from call_me_maybe.ui.spinner import spinner_title
from llm_sdk import Small_LLM_Model


class Generator:
    """Streams an LLM reply into the terminal with a typing animation."""

    MAX_TOKENS = 300

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
        text = Text(f"Opening model {model}", style="cyan")
        spinner = Spinner("dots", text=text, style="cyan")
        if self.__output:
            self.__output.renderables.append(spinner)
        try:
            self.llm = Small_LLM_Model(model)
        except OSError as e:
            text = Text(f"X {text}", style="red")
            raise ModelNotLoaded(e)
        finally:
            if self.__output:
                self.__output.renderables.remove(spinner)
                self.__output.renderables.append(text)

    async def run_prompt(
        self, prompt: Prompt, group: Group, attempt: int = 0
    ) -> tuple[str | None, Panel]:
        """Animate the prompt and its answer into `group`.

        Returns the answer, and the panel it was typed into, so that
        the caller can mark that panel as accepted or refused. If the
        grammar deadlocks (no token fits), the panel turns red and the
        answer is None, so the caller can retry with another `attempt`.
        """
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        generate_task = asyncio.create_task(
            self.__generate_tokens(prompt, queue, attempt)
        )
        generate_task.add_done_callback(Generator.__retrieve_task_exception)
        user_panel = self.__append_panel(group, self.USER_PANEL_TITLE, "red")
        await self.__animate(
            f">> {prompt.user_prompt}",
            user_panel,
            delay=self.USER_PROMPT_DELAY,
        )
        user_panel.title = self.USER_PANEL_TITLE
        assistant_panel = self.__append_panel(
            group, self.ASSISTANT_PANEL_TITLE, "yellow"
        )
        await self.__stream_tokens(
            queue, assistant_panel, delay=self.ASSISTANT_DELAY
        )
        try:
            output = await generate_task
        except RuntimeError:
            assistant_panel.title = self.ASSISTANT_PANEL_TITLE
            assistant_panel.style = "red"
            return (None, assistant_panel)
        assistant_panel.title = self.ASSISTANT_PANEL_TITLE
        return (output, assistant_panel)

    @staticmethod
    def __append_panel(group: Group, title: str, style: str) -> Panel:
        """Add an empty panel to `group`, titled with a spinner.

        The caller types into it and puts the plain title back once
        the panel is done.
        """
        panel = Panel(
            "",
            title=spinner_title(title),
            title_align="left",
            expand=False,
            style=style,
        )
        group.renderables.append(panel)
        return panel

    async def __stream_tokens(
        self,
        queue: asyncio.Queue[str | None],
        panel: Panel,
        *,
        delay: float,
    ) -> None:
        """Animate every token from `queue` into `panel`."""
        while True:
            token = await queue.get()
            if token is None:
                break
            # Type at the delay while the model keeps up, then flush
            # the backlog with no delay: waiting would only fall
            # further behind
            delay_now = delay if queue.empty() else 0.0
            await self.__animate(token, panel, delay=delay_now)

    async def __animate(
        self,
        text: str,
        panel: Panel,
        *,
        delay: float = 0.0,
    ) -> None:
        """Type `text` into `panel` one character at a time."""
        typed = str(panel.renderable)
        for char in text:
            typed += char
            panel.renderable = Text(typed, style="default")
            if delay:
                await asyncio.sleep(delay)

    async def __generate_tokens(
        self,
        prompt: Prompt,
        queue: asyncio.Queue[str | None],
        attempt: int,
    ) -> str:
        """Generate tokens and put each one on `queue`, ending with None."""
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.to_thread(
                self.__generate_tokens_loop, loop, prompt, queue, attempt
            )
        finally:
            await queue.put(None)

    def __generate_tokens_loop(
        self,
        loop: asyncio.AbstractEventLoop,
        prompt: Prompt,
        queue: asyncio.Queue[str | None],
        attempt: int,
    ) -> str:
        """Type the constrained answer and return the JSON of the call.

        It runs in a worker thread so the model never holds up the
        animation. A queue belongs to the loop that created it, so each
        token is handed over with `call_soon_threadsafe`, the only way
        to reach that loop from here.
        """
        input_ids = self.llm.encode(str(prompt))[0].tolist()
        functions = prompt.function_definitions
        answer = ""
        for _ in range(self.MAX_TOKENS):
            logits = self.llm.get_logits_from_input_ids(input_ids)
            token_id = self.__get_valid_token(
                logits, answer, functions, attempt
            )
            if token_id is None:
                # Every candidate was rejected
                raise RuntimeError("No token fits the JSON structure")
            input_ids.append(token_id)
            token_decoded = self.llm.decode([token_id])
            answer += token_decoded
            loop.call_soon_threadsafe(queue.put_nowait, token_decoded)
            if JSONGrammar.is_closed(answer, functions):
                # The closing brace is out: nothing valid can follow.
                break
        else:
            # The loop ended without a break: the budget ran out.
            logger.warning(
                f"Cut at {self.MAX_TOKENS} tokens before the JSON closed"
            )
        return answer

    @staticmethod
    def __retrieve_task_exception(task: asyncio.Task[str]) -> None:
        """Consume a finished task's exception to silence the warning."""
        if not task.cancelled():
            task.exception()

    def __get_valid_token(
        self,
        logits: list[float],
        answer: str,
        functions: FunctionDefinitionList,
        attempt: int,
    ) -> int | None:
        """The id of a token that keeps the answer a valid prefix.

        Attempt 0 picks the best token. Each retry takes the next-best
        valid token instead, so a failed answer is not repeated
        identically; it falls back to the best when fewer candidates
        survive than the attempt asks for.
        """
        best: int | None = None
        skipped = 0
        for token_id in np.argsort(logits)[::-1]:
            token_decoded = self.llm.decode([int(token_id)])
            if not token_decoded or not token_decoded.strip():
                # Special tokens decode to "" and a blank token is not
                # part of the call
                continue
            if "\n" in token_decoded:
                # The tokenizer drops blank space, so the grammar never
                # sees a line break and cannot turn it down. A break
                # also has no place in the single line the answer asks
                # for, so it is turned down here.
                continue
            if JSONGrammar.valid_prefix(answer + token_decoded, functions):
                if best is None:
                    best = int(token_id)
                if skipped == attempt:
                    return int(token_id)
                skipped += 1
        return best
