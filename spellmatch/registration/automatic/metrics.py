import logging
from abc import ABC
from enum import Enum

import SimpleITK as sitk
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Metric(BaseModel, ABC):
    sampling_percentage: float = 1.0
    sampling_strategy: str = "NONE"
    sampling_seed: int = sitk.sitkWallClock

    def configure(self, r: sitk.ImageRegistrationMethod) -> None:
        r.SetMetricSamplingPercentage(self.sampling_percentage, seed=self.sampling_seed)
        r.SetMetricSamplingStrategy(_SamplingStrategy[self.sampling_strategy].value)


class ANTSNeighborhoodCorrelationMetric(Metric):
    radius: int

    def configure(self, r: sitk.ImageRegistrationMethod) -> None:
        super(ANTSNeighborhoodCorrelationMetric, self).configure(r)
        r.SetMetricAsANTSNeighborhoodCorrelation(self.radius)


class CorrelationMetric(Metric):
    def configure(self, r: sitk.ImageRegistrationMethod) -> None:
        super(CorrelationMetric, self).configure(r)
        r.SetMetricAsCorrelation()


class DemonsMetric(Metric):
    intensity_diff_thres: float = 0.001

    def configure(self, r: sitk.ImageRegistrationMethod) -> None:
        super(DemonsMetric, self).configure(r)
        r.SetMetricAsDemons(intensityDifferenceThreshold=self.intensity_diff_thres)


class JointHistogramMutualInformationMetric(Metric):
    bins: int = 20
    smoothing_var: float = 1.5

    def configure(self, r: sitk.ImageRegistrationMethod) -> None:
        super(JointHistogramMutualInformationMetric, self).configure(r)
        r.SetMetricAsJointHistogramMutualInformation(
            numberOfHistogramBins=self.bins,
            varianceForJointPDFSmoothing=self.smoothing_var,
        )


class MattesMutualInformationMetric(Metric):
    bins: int = 50

    def configure(self, r: sitk.ImageRegistrationMethod) -> None:
        super(MattesMutualInformationMetric, self).configure(r)
        r.SetMetricAsMattesMutualInformation(numberOfHistogramBins=self.bins)


class MeanSquaresMetric(Metric):
    def configure(self, r: sitk.ImageRegistrationMethod) -> None:
        super(MeanSquaresMetric, self).configure(r)
        r.SetMetricAsMeanSquares()


class _SamplingStrategy(Enum):
    NONE = sitk.ImageRegistrationMethod.NONE
    REGULAR = sitk.ImageRegistrationMethod.REGULAR
    RANDOM = sitk.ImageRegistrationMethod.RANDOM
