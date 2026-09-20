from pathlib import Path
from string import Template

PROMPT_TEMPLATE_PATH = Path(__file__).parent / "templates" / "prompt.txt"
SYSTEM_PROMPT_DEFAULT_PATH = Path(__file__).parent / "templates" / "system.md"
FN_DEFINITIONS_DEFAULT_PATH: Path = Path(
    "data/input/functions_definition.json"
)
INPUT_DEFAULT_PATH: Path = Path("data/input/function_calling_tests.json")


def get_system_prompt_from_file(system_prompt_file: Path | None = None) -> str:
    if system_prompt_file:
        try:
            with open(system_prompt_file, "r") as file:
                text = file.read()
            return text
        except FileNotFoundError:
            print(
                "Error opening user defined system prompt file, \
                    using default system prompt"
            )
    with open(SYSTEM_PROMPT_DEFAULT_PATH, "r") as file:
        text = file.read()
    return text


def get_prompt_template(
    system_prompt: str, functions_definition: str
) -> Template:
    with open(PROMPT_TEMPLATE_PATH, "r") as file:
        template = Template(file.read())
    template.substitute(system=system_prompt, definitions=functions_definition)
    return template
