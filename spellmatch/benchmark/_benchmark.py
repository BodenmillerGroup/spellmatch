import logging

from .._spellmatch import SpellmatchError

logger = logging.getLogger(__name__)


class SpellmatchBenchmarkError(SpellmatchError):
    pass


# TODO benchmarks
