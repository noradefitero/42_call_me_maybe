.PHONY: install
install: ## Install the virtual environment and install the pre-commit hooks
	@uv sync
	@uv run pre-commit install

.PHONY: run
run: ## Run the program with default values
	@uv run python -m src

.PHONY: debug
debug: ## Run the Python debugger
	@uv run python -m pdb -m src

.PHONY: clean
clean: ## Clean Python artifacts
	@rm -rf $(foreach d,$(ARTIFACTS),$(shell find . -type d -name "$(d)"))

.PHONY: lint
lint: ## Run code quality tools
	@uv run flake8 .
	@uv run mypy . \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs \

.PHONY: test
test: ## Test the code with pytest
	@echo "🚀 Testing code: Running pytest"
	@uv run python -m pytest --cov --cov-config=pyproject.toml --cov-report=xml

.PHONY: help
help:
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help
