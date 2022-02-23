from typing import TYPE_CHECKING, Optional, Type

import pluggy

if TYPE_CHECKING:
    from .matching.algorithms import MatchingAlgorithm

hookspec = pluggy.HookspecMarker("spellmatch")


@hookspec(firstresult=True)
def spellmatch_get_matching_algorithm(name: str) -> Optional[Type["MatchingAlgorithm"]]:
    pass
