from typing import TYPE_CHECKING

import pluggy

if TYPE_CHECKING:
    from .matching.algorithms import MaskMatchingAlgorithm

hookspec = pluggy.HookspecMarker("spellmatch")


@hookspec
def spellmatch_get_mask_matching_algorithm(
    name: str | None,
) -> type["MaskMatchingAlgorithm"] | None | list[str]:
    pass
