try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._spellmatch import SpellmatchException, hookimpl, logger

__all__ = ["SpellmatchException", "hookimpl", "logger"]
