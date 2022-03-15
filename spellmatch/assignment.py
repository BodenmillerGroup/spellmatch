from enum import Enum
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import linear_sum_assignment
from skimage.color import label2rgb
from skimage.measure import regionprops
from skimage.segmentation import relabel_sequential
from skimage.util import img_as_ubyte

from spellmatch.matching.algorithms.spellmatch import SpellmatchException

from .utils import show_image


class AssignmentStrategy(Enum):
    THRESHOLD = "threshold"
    LINEAR_SUM = "linear_sum"
    FORWARD_MAX = "forward_max"
    REVERSE_MAX = "reverse_max"


assignment_strategies: dict[str, AssignmentStrategy] = {
    "threshold": AssignmentStrategy.THRESHOLD,
    "linear_sum": AssignmentStrategy.LINEAR_SUM,
    "forward_max": AssignmentStrategy.FORWARD_MAX,
    "reverse_max": AssignmentStrategy.REVERSE_MAX,
}


def assign(
    scores: xr.DataArray,
    strategy: AssignmentStrategy,
    normalize_directed: bool = False,
    directed_margin_thres: Optional[float] = None,
    score_thres: float = 0,
):
    scores_arr = scores.to_numpy()
    if normalize_directed:
        max_scores = np.amax(scores_arr, axis=1)
        scores_arr[max_scores != 0, :] /= np.sum(
            scores_arr[max_scores != 0, :], axis=1, keepdims=True
        )
    if directed_margin_thres is not None:
        max2_scores = -np.partition(-scores_arr, 1, axis=1)[:, :2]
        margins = max2_scores[:, 0] - max2_scores[:, 1]
        scores_arr[margins <= directed_margin_thres, :] = 0
    if strategy == AssignmentStrategy.THRESHOLD:
        if score_thres is None:
            raise SpellmatchAssignmentException("Unspecified score threshold")
        source_ind, target_ind = np.nonzero(scores_arr > score_thres)
    elif strategy == AssignmentStrategy.LINEAR_SUM:
        source_ind, target_ind = linear_sum_assignment(scores_arr, maximize=True)
    elif strategy == AssignmentStrategy.FORWARD_MAX:
        source_ind = np.arange(scores_arr.shape[0])
        target_ind = np.argmax(scores_arr, axis=1)
    elif strategy == AssignmentStrategy.REVERSE_MAX:
        source_ind = np.argmax(scores_arr, axis=0)
        target_ind = np.arange(scores_arr.shape[1])
    if score_thres is not None and strategy != AssignmentStrategy.THRESHOLD:
        m = scores_arr[source_ind, target_ind] > score_thres
        source_ind, target_ind = source_ind[m], target_ind[m]
    assignment = pd.DataFrame(
        data={
            "Source": scores.coords[scores.dims[0]].to_numpy()[source_ind],
            "Target": scores.coords[scores.dims[1]].to_numpy()[target_ind],
        }
    )
    return assignment


class AssigmentCombinationStrategy(Enum):
    UNION = "outer"
    INTERSECT = "inner"
    FORWARD_ONLY = "left"
    REVERSE_ONLY = "right"


assignment_combination_strategies: dict[str, AssigmentCombinationStrategy] = {
    "union": AssigmentCombinationStrategy.UNION,
    "intersect": AssigmentCombinationStrategy.INTERSECT,
    "forward_only": AssigmentCombinationStrategy.FORWARD_ONLY,
    "reverse_only": AssigmentCombinationStrategy.REVERSE_ONLY,
}


def combine_assignments(
    forward_assignment: pd.DataFrame,
    reverse_assignment: pd.DataFrame,
    strategy: AssigmentCombinationStrategy,
) -> pd.DataFrame:
    return pd.merge(
        forward_assignment,
        reverse_assignment,
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


class SpellmatchAssignmentException(SpellmatchException):
    pass
