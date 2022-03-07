import logging
from typing import Optional, Type, TypeVar, Union

import numpy as np
import pyqtgraph as pg
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
    "rigid": sitk.Euler2DTransform,
    "similarity": sitk.Similarity2DTransform,
    "affine": sitk.AffineTransform,
}


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
    visualize: bool = False,
) -> ProjectiveTransform:
    moving_img = sitk.GetImageFromArray(source_img.astype(float))
    moving_origin = (
        -0.5 * source_img.shape[-1] + 0.5,
        -0.5 * source_img.shape[-2] + 0.5,
    )
    if "scale" in source_img.attrs:
        moving_origin = tuple(x * source_img.attrs["scale"] for x in moving_origin)
        moving_img.SetSpacing((source_img.attrs["scale"], source_img.attrs["scale"]))
    moving_img.SetOrigin(moving_origin)
    if denoise_source is not None:
        moving_median_filter = sitk.MedianImageFilter()
        moving_median_filter.SetRadius(denoise_source)
        moving_img = moving_median_filter.Execute(moving_img)
    if blur_source is not None:
        moving_gaussian_filter = sitk.SmoothingRecursiveGaussianImageFilter()
        moving_gaussian_filter.SetNormalizeAcrossScale(True)
        moving_gaussian_filter.SetSigma(blur_source)
        moving_img = moving_gaussian_filter.Execute(moving_img)
    fixed_img = sitk.GetImageFromArray(target_img.astype(float))
    fixed_origin = (
        -0.5 * target_img.shape[-1] + 0.5,
        -0.5 * target_img.shape[-2] + 0.5,
    )
    if "scale" in target_img.attrs:
        fixed_origin = tuple(x * target_img.attrs["scale"] for x in fixed_origin)
        fixed_img.SetSpacing((target_img.attrs["scale"], target_img.attrs["scale"]))
    fixed_img.SetOrigin(fixed_origin)
    if denoise_target is not None:
        fixed_median_filter = sitk.MedianImageFilter()
        fixed_median_filter.SetRadius(denoise_target)
        fixed_img = fixed_median_filter.Execute(fixed_img)
    if blur_target is not None:
        fixed_gaussian_filter = sitk.SmoothingRecursiveGaussianImageFilter()
        fixed_gaussian_filter.SetNormalizeAcrossScale(True)
        fixed_gaussian_filter.SetSigma(blur_target)
        fixed_img = fixed_gaussian_filter.Execute(fixed_img)
    method = sitk.ImageRegistrationMethod()
    metric.configure(method)
    optimizer.configure(method)
    if initial_transform is not None:
        sitk_transform = _from_transform(initial_transform, sitk_transform_type)
    else:
        sitk_transform = sitk_transform_type()
    method.SetInitialTransform(sitk_transform, inPlace=True)
    method.SetInterpolator(sitk.sitkLinear)
    method.AddCommand(sitk.sitkIterationEvent, lambda: _log_on_iteration(method))
    if visualize:
        composite_img_arrs = [_compose_images(moving_img, fixed_img, sitk_transform)]
        method.AddCommand(
            sitk.sitkIterationEvent,
            lambda: composite_img_arrs.append(
                _compose_images(moving_img, fixed_img, sitk_transform)
            ),
        )
    method.Execute(fixed_img, moving_img)
    if visualize:
        pg.image(np.array(composite_img_arrs))
        pg.exec()
    return _to_transform(sitk_transform, ProjectiveTransform)


def _log_on_iteration(method: sitk.ImageRegistrationMethod) -> None:
    optimizer_iteration = method.GetOptimizerIteration()
    optimizer_position = method.GetOptimizerPosition()
    metric_value = method.GetMetricValue()
    logger.info(
        f"Iteration {optimizer_iteration:03}: {metric_value:.6f} {optimizer_position}"
    )


def _compose_images(
    moving_img: sitk.Image, fixed_img: sitk.Image, sitk_transform: sitk.Transform
) -> np.ndarray:
    resampled_img = sitk.Resample(moving_img, fixed_img, transform=sitk_transform)
    composite_img = sitk.Compose(resampled_img, fixed_img, resampled_img)
    return sitk.GetArrayFromImage(composite_img)


SITKTransformType = TypeVar("SITKTransformType", bound=SITKProjectiveTransform)


def _from_transform(
    transform: ProjectiveTransform, sitk_transform_type: Type[SITKTransformType]
) -> SITKTransformType:
    sitk_transform = sitk_transform_type()
    sitk_transform.SetTranslation(transform.params[:2, 2])
    sitk_transform.SetMatrix(transform.params[:2, :2].flatten())
    return sitk_transform


TransformType = TypeVar("TransformType", bound=ProjectiveTransform)


def _to_transform(
    sitk_transform: SITKProjectiveTransform, transform_type: Type[TransformType]
) -> TransformType:
    transform_matrix = np.eye(3)
    transform_matrix[:2, 2] = sitk_transform.GetTranslation()
    transform_matrix[:2, :2] = np.reshape(sitk_transform.GetMatrix(), (2, 2))
    return transform_type(matrix=transform_matrix)
