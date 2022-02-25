from typing import Optional, Type

from ..._spellmatch import hookimpl
from ._algorithms import MaskMatchingAlgorithm, IterativeGraphMatchingAlgorithm


# TODO implement spellmatch algorithm


@hookimpl
def spellmatch_get_mask_matching_algorithm(
    name: str,
) -> Optional[Type[MaskMatchingAlgorithm]]:
    if name == "spellmatch":
        return Spellmatch
    return None


class Spellmatch(IterativeGraphMatchingAlgorithm):
    pass
