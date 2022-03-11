from enum import IntEnum
from typing import Any, Optional

import cv2
import numpy as np
import pyqtgraph as pg
import xarray as xr
from qtpy.QtCore import QEventLoop, Qt
from skimage.transform import ProjectiveTransform

try:
    from cv2 import xfeatures2d as cv2_xfeatures2d
except ImportError:
    cv2_xfeatures2d = None


class MatcherType(IntEnum):
    FLANNBASED = cv2.DESCRIPTOR_MATCHER_FLANNBASED
    BRUTEFORCE = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE
    BRUTEFORCE_L1 = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_L1
    BRUTEFORCE_HAMMING = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    BRUTEFORCE_HAMMINGLUT = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMINGLUT
    BRUTEFORCE_SL2 = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_SL2


matcher_types: dict[str, MatcherType] = {
    "flannbased": MatcherType.FLANNBASED,
    "bruteforce": MatcherType.BRUTEFORCE,
    "bruteforce_l1": MatcherType.BRUTEFORCE_L1,
    "bruteforce_hamming": MatcherType.BRUTEFORCE_HAMMING,
    "bruteforce_hamminglut": MatcherType.BRUTEFORCE_HAMMINGLUT,
    "bruteforce_sl2": MatcherType.BRUTEFORCE_SL2,
}

# TODO add further features
feature_types: dict[str, callable] = {
    "ORB": cv2.ORB_create,
    "SIFT": cv2.SIFT_create,
}
if cv2_xfeatures2d is not None:
    feature_types.update(
        {
            "SURF": cv2_xfeatures2d.SURF_create,
        }
    )


def register_image_features(
    source_img: xr.DataArray,
    target_img: xr.DataArray,
    feature_type: callable = cv2.SIFT_create,
    feature_kwargs: Optional[dict[str, Any]] = None,
    matcher_type: MatcherType = MatcherType.BRUTEFORCE,
    keep_matches_frac: Optional[float] = None,
    ransac_kwargs: Optional[dict[str, Any]] = None,
    show: bool = False,
) -> ProjectiveTransform:
    if feature_kwargs is None:
        feature_kwargs = {}
    source_img = source_img.to_numpy()
    target_img = target_img.to_numpy()
    img_min = min(np.amin(source_img), np.amin(target_img))
    img_max = max(np.amax(source_img), np.amax(target_img))
    source_img = (source_img - img_min) / (img_max - img_min)
    target_img = (target_img - img_min) / (img_max - img_min)
    source_img = (source_img * 255).astype(np.uint8)
    target_img = (target_img * 255).astype(np.uint8)
    feature: cv2.Feature2D = feature_type(**feature_kwargs)
    source_kps, source_descs = feature.detectAndCompute(source_img, None)
    target_kps, target_descs = feature.detectAndCompute(target_img, None)
    matcher: cv2.DescriptorMatcher = cv2.DescriptorMatcher_create(int(matcher_type))
    matches = matcher.match(source_descs, target_descs)
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
    h, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, **ransac_kwargs)
    if show:
        matches_img = cv2.drawMatches(
            source_img, source_kps, target_img, target_kps, matches, None
        )
        pg.setConfigOption("imageAxisOrder", "row-major")
        pg.mkQApp()
        imv = pg.ImageView()
        imv_loop = QEventLoop()
        imv.setImage(matches_img)
        imv.destroyed.connect(imv_loop.quit)
        imv.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        imv.setWindowTitle("spellmatch registration")
        imv.show()
        imv_loop.exec()
    return ProjectiveTransform(matrix=h)
