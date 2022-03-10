from enum import IntEnum
from typing import Any, Optional

import cv2
import numpy as np
import pyqtgraph as pg
import xarray as xr
from qtpy.QtCore import QEventLoop, Qt
from scipy.ndimage import gaussian_filter, median_filter
from skimage.transform import ProjectiveTransform
from skimage.util import img_as_ubyte


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


def register_image_features(
    source_img: xr.DataArray,
    target_img: xr.DataArray,
    orb_kwargs: Optional[dict[str, Any]] = None,
    matcher_type: MatcherType = MatcherType.BRUTEFORCE,
    keep_matches_frac: Optional[float] = None,
    ransac_kwargs: Optional[dict[str, Any]] = None,
    denoise_source: Optional[int] = None,
    denoise_target: Optional[int] = None,
    blur_source: Optional[float] = None,
    blur_target: Optional[float] = None,
    show: bool = False,
) -> ProjectiveTransform:
    if denoise_source is not None:
        source_img = median_filter(source_img, size=2 * denoise_source + 1)
    if blur_source is not None:
        source_img = gaussian_filter(source_img, sigma=blur_source)
    source_img = img_as_ubyte(source_img.to_numpy().astype(np.uint8))  # TODO
    if denoise_target is not None:
        target_img = median_filter(target_img, size=2 * denoise_target + 1)
    if blur_target is not None:
        target_img = gaussian_filter(target_img, sigma=blur_target)
    target_img = img_as_ubyte(target_img.to_numpy().astype(np.uint8))  # TODO
    orb: cv2.ORB = cv2.ORB_create(**orb_kwargs)
    source_kps, source_descs = orb.detectAndCompute(source_img, None)
    target_kps, target_descs = orb.detectAndCompute(target_img, None)
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
