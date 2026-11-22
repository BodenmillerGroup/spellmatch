from skimage.transform import AffineTransform, EuclideanTransform, ProjectiveTransform

from .._spellmatch import SpellmatchException


class SpellmatchRegistrationException(SpellmatchException):
    pass


def get_min_point_pairs(transform_type: type[ProjectiveTransform]) -> int:
    """Minimum number of point pairs that determine a 2D transform."""
    if issubclass(transform_type, EuclideanTransform):  # incl. similarity transforms
        return 2
    if issubclass(transform_type, AffineTransform):
        return 3
    return 4
