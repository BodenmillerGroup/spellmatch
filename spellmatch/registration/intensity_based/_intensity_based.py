import logging
from typing import Optional, Union

import numpy as np
import SimpleITK as sitk
import xarray as xr
from qtpy.QtCore import QObject, QThread, Signal
from skimage.transform import ProjectiveTransform

from ...utils import show_image
from .sitk_metrics import SITKMetric
from .sitk_optimizers import SITKOptimizer

SITKProjectiveTransform = Union[
    sitk.Euler2DTransform, sitk.Similarity2DTransform, sitk.AffineTransform
]

logger = logging.getLogger(__name__.rpartition(".")[0])

sitk_transform_types: dict[str, SITKProjectiveTransform] = {
    "rigid": sitk.Euler2DTransform,
    "similarity": sitk.Similarity2DTransform,
    "affine": sitk.AffineTransform,
}


def register_image_intensities(
    source_img: xr.DataArray,
    target_img: xr.DataArray,
    sitk_metric: SITKMetric,
    sitk_optimizer: SITKOptimizer,
    sitk_transform_type: type[SITKProjectiveTransform] = sitk.AffineTransform,
    initial_transform: Optional[ProjectiveTransform] = None,
    show: bool = False,
    hold: bool = False,
) -> ProjectiveTransform:
    if source_img.ndim != 2 or target_img.ndim != 2:
        raise NotImplementedError("3D/multi-channel registration is not supported")
    moving_img = sitk.GetImageFromArray(source_img.astype(float))
    moving_origin = (
        -0.5 * source_img.shape[-1] + 0.5,
        -0.5 * source_img.shape[-2] + 0.5,
    )
    if "scale" in source_img.attrs:
        moving_origin = (
            moving_origin[0] * source_img.attrs["scale"][-1],
            moving_origin[1] * source_img.attrs["scale"][-2],
        )
        moving_img.SetSpacing(
            (source_img.attrs["scale"][-1], source_img.attrs["scale"][-2])
        )
    moving_img.SetOrigin(moving_origin)

    fixed_img = sitk.GetImageFromArray(target_img.astype(float))
    fixed_origin = (
        -0.5 * target_img.shape[-1] + 0.5,
        -0.5 * target_img.shape[-2] + 0.5,
    )
    if "scale" in target_img.attrs:
        fixed_origin = (
            fixed_origin[0] * target_img.attrs["scale"][-1],
            fixed_origin[1] * target_img.attrs["scale"][-2],
        )
        fixed_img.SetSpacing(
            (target_img.attrs["scale"][-1], target_img.attrs["scale"][-2])
        )
    fixed_img.SetOrigin(fixed_origin)

    sitk_transform = sitk_transform_type()
    if initial_transform is not None:
        sitk_transform.SetTranslation(initial_transform.params[:2, 2])
        sitk_transform.SetMatrix(initial_transform.params[:2, :2].ravel())
        sitk_transform = sitk_transform_type(sitk_transform.GetInverse())

    method = sitk.ImageRegistrationMethod()
    sitk_metric.configure(method)
    sitk_optimizer.configure(method)
    method.SetInitialTransform(sitk_transform, inPlace=True)
    method.SetInterpolator(sitk.sitkLinear)
    method.AddCommand(sitk.sitkIterationEvent, lambda: _log_on_iteration(method))

    if show:
        composite_imgs = []

        def append_composite_image() -> None:
            update_current_index = imv.currentIndex == len(composite_imgs) - 1
            resampled_moving_img = sitk.Resample(
                moving_img, fixed_img, transform=sitk_transform
            )
            composite_img = sitk.Compose(
                resampled_moving_img, fixed_img, resampled_moving_img
            )
            composite_imgs.append(sitk.GetArrayFromImage(composite_img))
            imv.setImage(np.stack(composite_imgs))
            if update_current_index:
                imv.setCurrentIndex(len(composite_imgs) - 1)

        imv, imv_loop = show_image(None, window_title="spellmatch registration")
        append_composite_image()

        worker = _QSITKRegistrationWorker(moving_img, fixed_img, method)
        worker.iteration.connect(append_composite_image)
        worker.register_commands()

        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        if not hold:
            thread.finished.connect(imv.close)

        thread.start()
        imv_loop.exec()
        thread.wait()
        sitk_transform = sitk_transform_type(worker.sitk_transform)
    else:
        sitk_transform = sitk_transform_type(method.Execute(fixed_img, moving_img))

    inverse_transform_matrix = np.eye(3)
    inverse_transform_matrix[:2, 2] = np.asarray(sitk_transform.GetTranslation())
    inverse_transform_matrix[:2, :2] = np.reshape(sitk_transform.GetMatrix(), (2, 2))
    return ProjectiveTransform(matrix=np.linalg.inv(inverse_transform_matrix))


def _log_on_iteration(method: sitk.ImageRegistrationMethod) -> None:
    optimizer_iteration = method.GetOptimizerIteration()
    optimizer_position = method.GetOptimizerPosition()
    metric_value = method.GetMetricValue()
    logger.info(
        f"Iteration {optimizer_iteration:03d}: {metric_value:.9f} {optimizer_position}"
    )


class _QSITKRegistrationWorker(QObject):
    iteration = Signal()
    finished = Signal()

    def __init__(
        self,
        moving_img: sitk.Image,
        fixed_img: sitk.Image,
        method: sitk.ImageRegistrationMethod,
    ) -> None:
        super(_QSITKRegistrationWorker, self).__init__()
        self.moving_img = moving_img
        self.fixed_img = fixed_img
        self.method = method
        self.sitk_transform: Optional[sitk.Transform] = None

    def register_commands(self) -> None:
        self.method.AddCommand(sitk.sitkIterationEvent, self.iteration_command)

    def run(self) -> None:
        self.sitk_transform = self.method.Execute(self.fixed_img, self.moving_img)
        self.finished.emit()

    def iteration_command(self) -> None:
        self.iteration.emit()
