from typing import TYPE_CHECKING, Optional, Union

import pluggy

if TYPE_CHECKING:
    from .matching.algorithms import MaskMatchingAlgorithm

hookspec = pluggy.HookspecMarker("spellmatch")


@hookspec
def spellmatch_get_mask_matching_algorithm(
    name: Optional[str],
) -> Union[Optional[type["MaskMatchingAlgorithm"]], list[str]]:
    pass
