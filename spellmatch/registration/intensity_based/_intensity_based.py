import logging
from typing import Optional, Type, Union

import numpy as np
import pyqtgraph as pg
import SimpleITK as sitk
import xarray as xr
from qtpy.QtCore import QEventLoop, QObject, Qt, QThread, Signal
from skimage.transform import ProjectiveTransform

from .sitk_metrics import SITKMetric
from .sitk_optimizers import SITKOptimizer

SITKProjectiveTransform = Union[
    sitk.Euler2DTransform, sitk.Similarity2DTransform, sitk.AffineTransform
]

logger = logging.getLogger(__name__)

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
    sitk_transform_type: Type[SITKProjectiveTransform] = sitk.AffineTransform,
    initial_transform: Optional[ProjectiveTransform] = None,
    denoise_source: Optional[int] = None,
    denoise_target: Optional[int] = None,
    blur_source: Optional[float] = None,
    blur_target: Optional[float] = None,
    show: bool = False,
    hold: bool = False,
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
        pg.setConfigOption("imageAxisOrder", "row-major")
        pg.mkQApp()
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

        imv = pg.ImageView()
        imv_loop = QEventLoop()
        imv.destroyed.connect(imv_loop.quit)
        imv.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        imv.setWindowTitle("spellmatch registration")
        append_composite_image()
        imv.show()

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
        f"Iteration {optimizer_iteration:03}: {metric_value:.6f} {optimizer_position}"
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
