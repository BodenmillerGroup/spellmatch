import logging

import pluggy


logger = logging.getLogger(__name__.rpartition(".")[0])

hookimpl = pluggy.HookimplMarker("spellmatch")


class SpellmatchException(Exception):
    pass
