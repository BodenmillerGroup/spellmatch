try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._spellmatch import SpellmatchError, hookimpl, logger

__all__ = ["SpellmatchError", "hookimpl", "logger"]
