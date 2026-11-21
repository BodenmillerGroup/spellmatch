import logging

import pluggy

logger = logging.getLogger(__name__.rpartition(".")[0])

hookimpl = pluggy.HookimplMarker("spellmatch")


class SpellmatchException(Exception):
    pass


class SpellmatchMissingDependencyException(SpellmatchException, ImportError):
    pass


def missing_gui_dependency(e: ImportError) -> SpellmatchMissingDependencyException:
    if isinstance(e, SpellmatchMissingDependencyException):
        return SpellmatchMissingDependencyException(*e.args)
    return SpellmatchMissingDependencyException(
        f"This feature requires the optional GUI dependencies (missing: {e.name}), "
        "which can be installed using: pip install 'spellmatch[gui]'"
    )
