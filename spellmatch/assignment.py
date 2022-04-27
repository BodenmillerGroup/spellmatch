from enum import Enum
from typing import Optional, Union

import cv2
import numpy as np
import pandas as pd
import xarray as xr
from scipy import optimize
from skimage.color import label2rgb
from skimage.measure import regionprops
from skimage.segmentation import relabel_sequential
from skimage.util import img_as_ubyte

from ._spellmatch import SpellmatchException
from .utils import show_image


class AssignmentDirection(Enum):
    FORWARD = "forward"
    REVERSE = "reverse"
    INTERSECT = "intersect"
    UNION = "union"


def assign(
    scores: xr.DataArray,
    reverse_scores: Optional[xr.DataArray] = None,
    normalize: bool = False,
    min_score: Optional[float] = None,
    min_score_quantile: Optional[float] = None,
    margin_thres: Optional[float] = None,
    margin_thres_quantile: Optional[float] = None,
    max_assignment: bool = False,
    linear_sum_assignment: bool = False,
    min_post_assignment_score: Optional[float] = None,
    min_post_assignment_score_quantile: Optional[float] = None,
    assignment_direction: Union[str, AssignmentDirection] = AssignmentDirection.FORWARD,
    as_matrix: bool = False,
) -> Union[pd.DataFrame, xr.DataArray]:
    if isinstance(assignment_direction, str):
        assignment_direction = AssignmentDirection(assignment_direction)
    fwd_scores = None
    if assignment_direction != AssignmentDirection.REVERSE:
        fwd_scores = scores.to_numpy().copy()
    rev_scores = None
    if assignment_direction != AssignmentDirection.FORWARD:
        if reverse_scores is not None:
            if reverse_scores.shape != scores.shape:
                raise SpellmatchAssignmentException(
                    "Reverse scores shape does not match (forward) scores shape"
                )
            rev_scores = reverse_scores.to_numpy().copy()
        else:
            rev_scores = scores.to_numpy().copy()
    if normalize:
        if fwd_scores is not None:
            max_fwd_scores = np.amax(fwd_scores, axis=1)
            fwd_scores[max_fwd_scores > 0, :] /= np.sum(
                fwd_scores[max_fwd_scores > 0, :], axis=1, keepdims=True
            )
            del max_fwd_scores
        if rev_scores is not None:
            max_rev_scores = np.amax(rev_scores, axis=0)
            rev_scores[:, max_rev_scores > 0] /= np.sum(
                rev_scores[:, max_rev_scores > 0], axis=0, keepdims=True
            )
            del max_rev_scores
    if min_score is not None:
        if fwd_scores is not None:
            fwd_scores[fwd_scores < min_score] = 0
        if rev_scores is not None:
            rev_scores[rev_scores < min_score] = 0
    if min_score_quantile is not None:
        if fwd_scores is not None:
            max_fwd_scores = np.amax(fwd_scores, axis=1)
            min_max_fwd_score = np.quantile(max_fwd_scores, min_score_quantile)
            fwd_scores[max_fwd_scores < min_max_fwd_score, :] = 0
            del max_fwd_scores, min_max_fwd_score
        if rev_scores is not None:
            max_rev_scores = np.amax(rev_scores, axis=0)
            min_max_rev_score = np.quantile(max_rev_scores, min_score_quantile)
            rev_scores[:, max_rev_scores < min_max_rev_score] = 0
            del max_rev_scores, min_max_rev_score
    if margin_thres is not None or margin_thres_quantile is not None:
        if fwd_scores is not None:
            max2_fwd_scores = -np.partition(-fwd_scores, 1, axis=1)[:, :2]
            fwd_margins = max2_fwd_scores[:, 0] - max2_fwd_scores[:, 1]
            del max2_fwd_scores
        if rev_scores is not None:
            max2_rev_scores = -np.partition(-rev_scores, 1, axis=0)[:2, :]
            rev_margins = max2_rev_scores[0, :] - max2_rev_scores[1, :]
            del max2_rev_scores
        if margin_thres is not None:
            if fwd_scores is not None:
                fwd_scores[fwd_margins <= margin_thres, :] = 0
            if rev_scores is not None:
                rev_scores[:, rev_margins <= margin_thres] = 0
        if margin_thres_quantile is not None:
            if fwd_scores is not None:
                fwd_margin_thres = np.quantile(fwd_margins, margin_thres_quantile)
                fwd_scores[fwd_margins <= fwd_margin_thres, :] = 0
                del fwd_margin_thres
            if rev_scores is not None:
                rev_margin_thres = np.quantile(rev_margins, margin_thres_quantile)
                rev_scores[:, rev_margins <= rev_margin_thres] = 0
                del rev_margin_thres
    if max_assignment:
        if fwd_scores is not None:
            row_ind = np.arange(fwd_scores.shape[0])
            col_ind = np.argmax(fwd_scores, axis=1)
            new_fwd_scores = np.zeros_like(fwd_scores)
            new_fwd_scores[row_ind, col_ind] = fwd_scores[row_ind, col_ind]
            fwd_scores = new_fwd_scores
            del new_fwd_scores, row_ind, col_ind
        if rev_scores is not None:
            row_ind = np.argmax(rev_scores, axis=0)
            col_ind = np.arange(fwd_scores.shape[1])
            new_rev_scores = np.zeros_like(rev_scores)
            new_rev_scores[row_ind, col_ind] = rev_scores[row_ind, col_ind]
            rev_scores = new_rev_scores
            del new_rev_scores, row_ind, col_ind
    if linear_sum_assignment:
        if fwd_scores is not None:
            row_ind, col_ind = optimize.linear_sum_assignment(fwd_scores, maximize=True)
            new_fwd_scores = np.zeros_like(fwd_scores)
            new_fwd_scores[row_ind, col_ind] = fwd_scores[row_ind, col_ind]
            fwd_scores = new_fwd_scores
            del new_fwd_scores, row_ind, col_ind
        if rev_scores is not None:
            row_ind, col_ind = optimize.linear_sum_assignment(rev_scores, maximize=True)
            new_rev_scores = np.zeros_like(rev_scores)
            new_rev_scores[row_ind, col_ind] = rev_scores[row_ind, col_ind]
            rev_scores = new_rev_scores
            del new_rev_scores, row_ind, col_ind
    if min_post_assignment_score is not None:
        if fwd_scores is not None:
            fwd_scores[fwd_scores < min_post_assignment_score] = 0
        if rev_scores is not None:
            rev_scores[rev_scores < min_post_assignment_score] = 0
    if min_post_assignment_score_quantile is not None:
        if fwd_scores is not None:
            min_fwd_score = np.quantile(
                fwd_scores[fwd_scores > 0], min_post_assignment_score_quantile
            )
            fwd_scores[fwd_scores < min_fwd_score] = 0
            del min_fwd_score
        if rev_scores is not None:
            min_rev_score = np.quantile(
                rev_scores[rev_scores > 0], min_post_assignment_score_quantile
            )
            rev_scores[rev_scores < min_rev_score] = 0
            del min_rev_score
    if assignment_direction == AssignmentDirection.FORWARD:
        assignment_data = fwd_scores > 0
    elif assignment_direction == AssignmentDirection.REVERSE:
        assignment_data = rev_scores > 0
    elif assignment_direction == AssignmentDirection.INTERSECT:
        assignment_data = (fwd_scores > 0) & (rev_scores > 0)
    elif assignment_direction == AssignmentDirection.UNION:
        assignment_data = (fwd_scores > 0) | (rev_scores > 0)
    else:
        raise NotImplementedError()
    del fwd_scores, rev_scores
    if as_matrix:
        assignment_mat = scores.copy(data=assignment_data)
        return assignment_mat
    source_ind, target_ind = np.where(assignment_data)
    assignment = pd.DataFrame(
        data={
            scores.dims[0]: scores.coords[scores.dims[0]].to_numpy()[source_ind],
            scores.dims[1]: scores.coords[scores.dims[1]].to_numpy()[target_ind],
        }
    )
    del source_ind, target_ind
    return assignment


def validate_assignment(
    assignment: pd.DataFrame, validation_assignment: pd.DataFrame
) -> float:
    merged_assignment = pd.merge(assignment, validation_assignment)
    return len(merged_assignment.index) / len(validation_assignment.index)


def show_assignment(
    source_mask: xr.DataArray,
    target_mask: xr.DataArray,
    assignment: pd.DataFrame,
    n: int,
) -> None:
    filtered_source_mask = source_mask.to_numpy().copy()
    filtered_source_mask[
        ~np.isin(filtered_source_mask, assignment.iloc[:, 0].to_numpy())
    ] = 0
    filtered_target_mask = target_mask.to_numpy().copy()
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
