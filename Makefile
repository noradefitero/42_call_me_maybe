install:
	@uv sync

run:
	@uv run python -m src

init-repo:
	@pre-commit install