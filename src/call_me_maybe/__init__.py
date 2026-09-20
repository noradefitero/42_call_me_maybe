try:
    from importlib.metadata import version as _package_version

    __version__ = _package_version("call-me-maybe")
except ImportError:
    __version__ = "0.1.0"

from .main import app
