import asyncio

from call_me_maybe.generator import Generator
from call_me_maybe.models import FunctionCallList
from call_me_maybe.pipeline.parse import ParsedArgs
from call_me_maybe.prompt import Prompt


def run(args: ParsedArgs, model: str) -> FunctionCallList:
    """Run the function-calling pipeline for the given inputs."""
    response = FunctionCallList([])
    prompt = Prompt(
        template=args["template"],
        system=args["system_prompt"],
        definitions=args["definitions"],
        user_prompt=args["input"][0],
    )
    generator = Generator(model)
    asyncio.run(generator.run_prompt(prompt))
    return response
