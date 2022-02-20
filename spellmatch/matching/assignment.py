import logging

from ._matching import SpellmatchMatchingError

logger = logging.getLogger(__name__)


class SpellmatchMatchingAssignmentError(SpellmatchMatchingError):
    pass
