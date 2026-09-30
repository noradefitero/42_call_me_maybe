class PromptLoadError(Exception):
    """Raised when a prompt, definitions or input file is unreadable."""


class ModelNotLoaded(Exception):
    """Raised when the requested model cannot be opened."""


class OutputWriteError(Exception):
    """Raised when the results file cannot be written."""


class NoValidCallError(Exception):
    """Raised when a prompt gets no schema-valid call in any attempt."""
