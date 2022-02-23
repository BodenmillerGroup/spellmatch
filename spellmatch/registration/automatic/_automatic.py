import logging
from typing import Optional, Type, Union

import numpy as np
import SimpleITK as sitk
import xarray as xr

from .._registration import SpellmatchRegistrationError
from .metrics import Metric
from .optimizers import Optimizer

Transform = Union[
    sitk.Euler2DTransform, sitk.Similarity2DTransform, sitk.AffineTransform
]

logger = logging.getLogger(__name__)

transform_types: dict[str, Transform] = {
    "euclidean": sitk.Euler2DTransform,
    "similarity": sitk.Similarity2DTransform,
    "affine": sitk.AffineTransform,
}

# TODO automatic registation visualization


def register_images(
    source_img: xr.DataArray,
    target_img: xr.DataArray,
    metric: Metric,
    optimizer: Optimizer,
    transform_type: Type[Transform] = sitk.AffineTransform,
    initial_transform_matrix: Optional[np.ndarray] = None,
    denoise_source: Optional[float] = None,
    denoise_target: Optional[float] = None,
    blur_source: Optional[float] = None,
    blur_target: Optional[float] = None,
) -> np.ndarray:
    moving = sitk.GetImageFromArray(source_img.to_numpy())
    if "scale" in source_img.attrs:
        moving.SetSpacing((source_img.attrs["scale"], source_img.attrs["scale"]))
    if denoise_source is not None:
        median_filter = sitk.MedianImageFilter()
        median_filter.SetRadius(denoise_source)
        moving = median_filter.Execute(moving)
    if blur_source is not None:
        gaussian_filter = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian_filter.SetNormalizeAcrossScale(True)
        gaussian_filter.SetSigma(blur_source)
        moving = gaussian_filter.Execute(moving)
    fixed = sitk.GetImageFromArray(target_img)
    if "scale" in target_img.attrs:
        fixed.SetSpacing((target_img.attrs["scale"], target_img.attrs["scale"]))
    if denoise_target is not None:
        median_filter = sitk.MedianImageFilter()
        median_filter.SetRadius(denoise_target)
        fixed = median_filter.Execute(fixed)
    if blur_target is not None:
        gaussian_filter = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian_filter.SetNormalizeAcrossScale(True)
        gaussian_filter.SetSigma(blur_target)
        fixed = gaussian_filter.Execute(fixed)
    method = sitk.ImageRegistrationMethod()
    metric.configure(method)
    optimizer.configure(method)
    initial_transform: Transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        transform_type(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    if initial_transform_matrix is not None:
        initial_transform.SetMatrix(initial_transform_matrix.flatten())
    method.SetInitialTransform(initial_transform)
    method.SetInterpolator(sitk.sitkLinear)
    method.AddCommand(lambda: _logging_command(method))
    transform: Transform = method.Execute(fixed, moving)
    return np.asarray(transform.GetMatrix()).reshape((3, 3))


def _logging_command(method: sitk.ImageRegistrationMethod) -> None:
    optimizer_iteration = method.GetOptimizerIteration()
    optimizer_position = method.GetOptimizerPosition()
    metric_value = method.GetMetricValue()
    logger.info(
        f"{optimizer_iteration:03} = {metric_value:9.6f} : {optimizer_position}"
    )


class SpellmatchAutomaticRegistrationError(SpellmatchRegistrationError):
    pass
