from abc import ABC
from enum import Enum

import SimpleITK as sitk
from pydantic import BaseModel


class SITKMetric(BaseModel, ABC):
    sampling_percentage: float = 1
    sampling_strategy: str = "NONE"
    sampling_seed: int = sitk.sitkWallClock

    class SamplingStrategy(Enum):
        NONE = sitk.ImageRegistrationMethod.NONE
        REGULAR = sitk.ImageRegistrationMethod.REGULAR
        RANDOM = sitk.ImageRegistrationMethod.RANDOM

    def configure(self, r: sitk.ImageRegistrationMethod) -> None:
        r.SetMetricSamplingPercentage(self.sampling_percentage, seed=self.sampling_seed)
        r.SetMetricSamplingStrategy(
            SITKMetric.SamplingStrategy[self.sampling_strategy].value
        )


class ANTSNeighborhoodCorrelationSITKMetric(SITKMetric):
    radius: int

    def configure(self, r: sitk.ImageRegistrationMethod) -> None:
        super(ANTSNeighborhoodCorrelationSITKMetric, self).configure(r)
        r.SetMetricAsANTSNeighborhoodCorrelation(self.radius)


class CorrelationSITKMetric(SITKMetric):
    def configure(self, r: sitk.ImageRegistrationMethod) -> None:
        super(CorrelationSITKMetric, self).configure(r)
        r.SetMetricAsCorrelation()


class DemonsSITKMetric(SITKMetric):
    intensity_diff_thres: float = 0.001

    def configure(self, r: sitk.ImageRegistrationMethod) -> None:
        super(DemonsSITKMetric, self).configure(r)
        r.SetMetricAsDemons(intensityDifferenceThreshold=self.intensity_diff_thres)


class JointHistogramMutualInformationSITKMetric(SITKMetric):
    bins: int = 20
    smoothing_var: float = 1.5

    def configure(self, r: sitk.ImageRegistrationMethod) -> None:
        super(JointHistogramMutualInformationSITKMetric, self).configure(r)
        r.SetMetricAsJointHistogramMutualInformation(
            numberOfHistogramBins=self.bins,
            varianceForJointPDFSmoothing=self.smoothing_var,
        )


class MattesMutualInformationSITKMetric(SITKMetric):
    bins: int = 50

    def configure(self, r: sitk.ImageRegistrationMethod) -> None:
        super(MattesMutualInformationSITKMetric, self).configure(r)
        r.SetMetricAsMattesMutualInformation(numberOfHistogramBins=self.bins)


class MeanSquaresSITKMetric(SITKMetric):
    def configure(self, r: sitk.ImageRegistrationMethod) -> None:
        super(MeanSquaresSITKMetric, self).configure(r)
        r.SetMetricAsMeanSquares()


sitk_metric_types: dict[str, type[SITKMetric]] = {
    "ants_neighborhood_correlation": ANTSNeighborhoodCorrelationSITKMetric,
    "correlation": CorrelationSITKMetric,
    "demons": DemonsSITKMetric,
    "joint_histogram_mutual_information": JointHistogramMutualInformationSITKMetric,
    "mattes_mutual_information": MattesMutualInformationSITKMetric,
    "mean_squares": MeanSquaresSITKMetric,
}
