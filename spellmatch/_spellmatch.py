import logging

import pluggy


logger = logging.getLogger(__name__)

hookimpl = pluggy.HookimplMarker("spellmatch")


class SpellmatchException(Exception):
    pass