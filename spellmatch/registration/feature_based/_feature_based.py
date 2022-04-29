from collections.abc import Mapping
from enum import Enum
from typing import Any, Optional

import cv2
import numpy as np
import xarray as xr
from skimage.transform import ProjectiveTransform

from ...utils import show_image

try:
    from cv2 import xfeatures2d as cv2_xfeatures2d  # type: ignore
except ImportError:
    cv2_xfeatures2d = None


class CV2MatcherType(Enum):
    FLANNBASED = cv2.DESCRIPTOR_MATCHER_FLANNBASED
    BRUTEFORCE = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE
    BRUTEFORCE_L1 = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_L1
    BRUTEFORCE_HAMMING = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    BRUTEFORCE_HAMMINGLUT = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMINGLUT
    BRUTEFORCE_SL2 = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_SL2


cv2_matcher_types: dict[str, CV2MatcherType] = {
    "flannbased": CV2MatcherType.FLANNBASED,
    "bruteforce": CV2MatcherType.BRUTEFORCE,
    "bruteforce_l1": CV2MatcherType.BRUTEFORCE_L1,
    "bruteforce_hamming": CV2MatcherType.BRUTEFORCE_HAMMING,
    "bruteforce_hamminglut": CV2MatcherType.BRUTEFORCE_HAMMINGLUT,
    "bruteforce_sl2": CV2MatcherType.BRUTEFORCE_SL2,
}

cv2_feature_types: dict[str, Any] = {
    "ORB": cv2.ORB_create,
    "SIFT": cv2.SIFT_create,
}
if cv2_xfeatures2d is not None:
    cv2_feature_types.update(
        {
            "SURF": cv2_xfeatures2d.SURF_create,
        }
    )


def register_image_features(
    source_img: xr.DataArray,
    target_img: xr.DataArray,
    cv2_feature: cv2.Feature2D,
    cv2_matcher_type: CV2MatcherType = CV2MatcherType.BRUTEFORCE,
    keep_matches_frac: Optional[float] = None,
    ransac_kwargs: Optional[Mapping[str, Any]] = None,
    show: bool = False,
) -> ProjectiveTransform:
    if source_img.ndim != 2 or target_img.ndim != 2:
        raise NotImplementedError("3D/multi-channel registration is not supported")
    source_img = source_img.to_numpy()
    target_img = target_img.to_numpy()
    img_min = min(np.amin(source_img), np.amin(target_img))
    img_max = max(np.amax(source_img), np.amax(target_img))
    source_img = (source_img - img_min) / (img_max - img_min)
    target_img = (target_img - img_min) / (img_max - img_min)
    source_img = (source_img * 255).astype(np.uint8)
    target_img = (target_img * 255).astype(np.uint8)
    source_kps, source_descs = cv2_feature.detectAndCompute(source_img, None)
    target_kps, target_descs = cv2_feature.detectAndCompute(target_img, None)
    cv2_matcher = cv2.DescriptorMatcher_create(cv2_matcher_type.value)
    matches = cv2_matcher.match(source_descs, target_descs)
    if keep_matches_frac is not None:
        matches = matches[: int(len(matches) * keep_matches_frac)]
    if len(matches) == 0:
        return None
    src = np.empty((len(matches), 2))
    dst = np.empty((len(matches), 2))
    for i, match in enumerate(matches):
        src[i] = source_kps[match.queryIdx].pt
        dst[i] = target_kps[match.trainIdx].pt
    src += -0.5 * np.array([source_img.shape]) + 0.5
    dst += -0.5 * np.array([target_img.shape]) + 0.5
    h, _ = cv2.findHomography(src, dst, method=cv2.RANSAC, **ransac_kwargs)
    if show:
        matches_img = cv2.drawMatches(
            source_img, source_kps, target_img, target_kps, matches, None
        )
        _, imv_loop = show_image(matches_img, window_title="spellmatch registration")
        imv_loop.exec()
    return ProjectiveTransform(matrix=h)
