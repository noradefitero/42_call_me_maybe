from pathlib import Path

# from llm_sdk import Small_LLM_Model
import typer

from call_me_maybe import __version__
from call_me_maybe.arguments import Arguments
from call_me_maybe.exception import NotInitiatedContext
from call_me_maybe.prompt import (
    FN_DEFINITIONS_DEFAULT_PATH,
    INPUT_DEFAULT_PATH,
    get_system_prompt_from_file,
)
from call_me_maybe.schema.fn_list import FnList

arguments: Arguments
app = typer.Typer()


def version_callback(value: bool) -> None:
    if value:
        print(f"Awesome CLI Version: {__version__}")
        raise typer.Exit()


@app.command()
def main(
    functions_definition: Path = FN_DEFINITIONS_DEFAULT_PATH,
    input: Path = INPUT_DEFAULT_PATH,
    output: Path = Path("data/output/function_calls.json"),
    system_prompt: str | None = None,
    system_prompt_file: Path | None = None,
    model: str = "Qwen/Qwen3-0.6B",
) -> None:

    if system_prompt is None:
        try:
            system_prompt = get_system_prompt_from_file(system_prompt_file)
        except FileNotFoundError:
            raise NotInitiatedContext("Failed loading system prompt")
    with open(functions_definition) as file:
        fn_list = FnList.model_validate_json(file.read())
    print(fn_list)
    # template =
    # arguments = Arguments(
    #     functions_definition=functions_definition,
    #     input=input,
    #     output=output,
    #     system_prompt=system_prompt,
    #     model=model,
    # )

    # llm = Small_LLM_Model(args.model)
    # prompt = ""
    # system_prompt

    # # if args.system_prompt:

    # with open(args.input, "r") as file:
    #     text = file.read()
    #     tokenized_input = llm.encode(text)
    #     llm.get_logits_from_input_ids(tokenized_input)
    #     tokenized_output = llm.get_logits_from_input_ids(tokenized_input)
    #     print(llm.decode(tokenized_output))


if __name__ == "main":
    app()
