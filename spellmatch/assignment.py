from enum import Enum
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import xarray as xr
from skimage.color import label2rgb
from skimage.measure import regionprops
from skimage.segmentation import relabel_sequential
from skimage.util import img_as_ubyte

from .utils import show_image


def assign_scores(
    scores: xr.DataArray,
    normalize: bool = False,
    score_thres: Optional[float] = None,
    margin_thres: Optional[float] = None,
) -> pd.DataFrame:
    if normalize:
        scores = scores.copy()
        score_sums = np.sum(scores.to_numpy(), axis=1)
        scores[score_sums != 0] /= score_sums[score_sums != 0, np.newaxis]
    source_ind = np.arange(scores.shape[0])
    max2_score_ind = np.argpartition(-scores.to_numpy(), 1, axis=1)[:, :2]
    max2_scores = np.take_along_axis(scores.to_numpy(), max2_score_ind, axis=1)
    if score_thres is not None:
        m = max2_scores[:, 0] > score_thres
        source_ind = source_ind[m]
        max2_score_ind = max2_score_ind[m, :]
        max2_scores = max2_scores[m, :]
    if margin_thres is not None:
        margins = max2_scores[:, 0] - max2_scores[:, 1]
        m = margins > margin_thres
        source_ind = source_ind[m]
        max2_score_ind = max2_score_ind[m, :]
        max2_scores = max2_scores[m, :]
    target_ind = max2_score_ind[:, 0]
    assignment = pd.DataFrame(
        data={
            "Source": scores.coords[scores.dims[0]].to_numpy()[source_ind],
            "Target": scores.coords[scores.dims[1]].to_numpy()[target_ind],
        }
    )
    return assignment


class AssigmentCombinationStrategy(Enum):
    UNION = "outer"
    INTERSECTION = "inner"
    FORWARD_ONLY = "left"
    REVERSE_ONLY = "right"


assignment_combination_strategies: dict[str, AssigmentCombinationStrategy] = {
    "union": AssigmentCombinationStrategy.UNION,
    "intersection": AssigmentCombinationStrategy.INTERSECTION,
    "forward_only": AssigmentCombinationStrategy.FORWARD_ONLY,
    "reverse_only": AssigmentCombinationStrategy.REVERSE_ONLY,
}


def combine_assignments(
    forward_assignment: pd.DataFrame,
    reverse_assignment: pd.DataFrame,
    strategy: AssigmentCombinationStrategy = AssigmentCombinationStrategy.INTERSECTION,
) -> pd.DataFrame:
    reversed_reverse_assigmnent = reverse_assignment.copy()
    reversed_reverse_assigmnent["Source"], reversed_reverse_assigmnent["Target"] = (
        reversed_reverse_assigmnent["Target"],
        reversed_reverse_assigmnent["Source"],
    )
    return pd.merge(
        forward_assignment,
        reversed_reverse_assigmnent,
        how=strategy.value,
        on=["Source", "Target"],
    )


def validate_assignment(
    assignment: pd.DataFrame, validation_assignment: pd.DataFrame
) -> float:
    merged_assignment = pd.merge(
        assignment, validation_assignment, how="inner", on=["Source", "Target"]
    )
    return len(merged_assignment.index) / len(validation_assignment.index)


def show_assignment(
    source_mask: xr.DataArray,
    target_mask: xr.DataArray,
    assignment: pd.DataFrame,
    n: int,
) -> None:
    filtered_source_mask = source_mask.to_numpy()
    filtered_source_mask[
        ~np.isin(filtered_source_mask, assignment.iloc[:, 0].to_numpy())
    ] = 0
    filtered_target_mask = target_mask.to_numpy()
    filtered_target_mask[
        ~np.isin(filtered_target_mask, assignment.iloc[:, 1].to_numpy())
    ] = 0
    relabeled_filtered_source_mask, fw, _ = relabel_sequential(filtered_source_mask)
    lut = np.zeros(fw.out_values.max() + 1, dtype=fw.out_values.dtype)
    lut[fw[assignment.iloc[:, 0].to_numpy()]] = assignment.iloc[:, 1].to_numpy()
    matched_filtered_source_mask = lut[relabeled_filtered_source_mask]
    img1 = img_as_ubyte(label2rgb(matched_filtered_source_mask)[:, :, ::-1])
    img2 = img_as_ubyte(label2rgb(filtered_target_mask)[:, :, ::-1])
    source_regions = regionprops(filtered_source_mask)
    target_regions = regionprops(filtered_target_mask)
    source_indices = {region.label: i for i, region in enumerate(source_regions)}
    target_indices = {region.label: i for i, region in enumerate(target_regions)}
    keypoints1 = [
        cv2.KeyPoint(region.centroid[-1], region.centroid[-2], 1.0)
        for region in source_regions
    ]
    keypoints2 = [
        cv2.KeyPoint(region.centroid[-1], region.centroid[-2], 1.0)
        for region in target_regions
    ]
    matches = [
        cv2.DMatch(source_indices[source_label], target_indices[target_label], 0.0)
        for source_label, target_label in assignment.sample(n=n).to_numpy()
    ]
    matches_img = cv2.drawMatches(
        img1,
        keypoints1,
        img2,
        keypoints2,
        matches,
        None,
    )
    _, imv_loop = show_image(matches_img, window_title="spellmatch assignment")
    imv_loop.exec()
