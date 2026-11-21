import SimpleITK as sitk

from ..._spellmatch import missing_gui_dependency

try:
    from qtpy.QtCore import QObject, Signal
except ImportError as e:
    raise missing_gui_dependency(e) from e


class QSITKRegistrationWorker(QObject):
    iteration = Signal()
    finished = Signal()

    def __init__(
        self,
        moving_img: sitk.Image,
        fixed_img: sitk.Image,
        method: sitk.ImageRegistrationMethod,
    ) -> None:
        super().__init__()
        self.moving_img = moving_img
        self.fixed_img = fixed_img
        self.method = method
        self.sitk_transform: sitk.Transform | None = None

    def register_commands(self) -> None:
        self.method.AddCommand(sitk.sitkIterationEvent, self.iteration_command)

    def run(self) -> None:
        self.sitk_transform = self.method.Execute(self.fixed_img, self.moving_img)
        self.finished.emit()

    def iteration_command(self) -> None:
        self.iteration.emit()
