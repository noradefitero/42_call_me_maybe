from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"

DEFAULT_FUNCTIONS_FILE = INPUT_DIR / "functions_definition.json"
DEFAULT_INPUT_FILE = INPUT_DIR / "function_calling_tests.json"
DEFAULT_OUTPUT_FILE = OUTPUT_DIR / "function_calls.json"

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
SYSTEM_PROMPT_DEFAULT_PATH = TEMPLATES_DIR / "system.md"
PROMPT_TEMPLATE_PATH = TEMPLATES_DIR / "prompt.txt"

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
