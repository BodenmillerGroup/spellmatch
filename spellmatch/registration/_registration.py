import logging
from typing import Optional, Type, Union

import numpy as np
import SimpleITK as sitk
import xarray as xr
from skimage.transform import ProjectiveTransform

from .metrics import Metric
from .optimizers import Optimizer

SITKProjectiveTransform = Union[
    sitk.Euler2DTransform, sitk.Similarity2DTransform, sitk.AffineTransform
]

logger = logging.getLogger(__name__)

sitk_transform_types: dict[str, SITKProjectiveTransform] = {
    "euclidean": sitk.Euler2DTransform,
    "similarity": sitk.Similarity2DTransform,
    "affine": sitk.AffineTransform,
}


# TODO registration logging
# TODO registration visualization


def register_images(
    source_img: xr.DataArray,
    target_img: xr.DataArray,
    metric: Metric,
    optimizer: Optimizer,
    sitk_transform_type: Type[SITKProjectiveTransform] = sitk.AffineTransform,
    initial_transform: Optional[ProjectiveTransform] = None,
    denoise_source: Optional[float] = None,
    denoise_target: Optional[float] = None,
    blur_source: Optional[float] = None,
    blur_target: Optional[float] = None,
) -> ProjectiveTransform:
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
    initial_sitk_transform: SITKProjectiveTransform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk_transform_type(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    if initial_transform is not None:
        initial_sitk_transform.SetMatrix(initial_transform.params.flatten())
    method.SetInitialTransform(initial_sitk_transform)
    method.SetInterpolator(sitk.sitkLinear)
    method.AddCommand(lambda: _logging_command(method))
    sitk_transform: SITKProjectiveTransform = method.Execute(fixed, moving)
    transform_matrix = np.asarray(sitk_transform.GetMatrix()).reshape((3, 3))
    return ProjectiveTransform(matrix=transform_matrix)


def _logging_command(method: sitk.ImageRegistrationMethod) -> None:
    optimizer_iteration = method.GetOptimizerIteration()
    optimizer_position = method.GetOptimizerPosition()
    metric_value = method.GetMetricValue()
    logger.info(
        f"{optimizer_iteration:03} = {metric_value:9.6f} : {optimizer_position}"
    )
