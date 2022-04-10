from abc import ABC
from collections.abc import Sequence
from enum import Enum
from typing import Optional, Union

import SimpleITK as sitk
from pydantic import BaseModel


class SITKOptimizer(BaseModel, ABC):
    scales: Union[str, Sequence[float], None] = None
    central_region_radius: int = 5
    small_parameter_variation: float = 0.01
    weights: Optional[Sequence[float]] = None

    class LearningRateEstimationType(Enum):
        Never = sitk.ImageRegistrationMethod.Never
        Once = sitk.ImageRegistrationMethod.Once
        EachIteration = sitk.ImageRegistrationMethod.EachIteration

    def configure(self, r: sitk.ImageRegistrationMethod) -> None:
        if self.scales is not None:
            if self.scales == "from_index_shift":
                r.SetOptimizerScalesFromIndexShift(
                    centralRegionRadius=self.central_region_radius,
                    smallParameterVariation=self.small_parameter_variation,
                )
            elif self.scales == "from_physical_shift":
                r.SetOptimizerScalesFromPhysicalShift(
                    centralRegionRadius=self.central_region_radius,
                    smallParameterVariation=self.small_parameter_variation,
                )
            elif self.scales == "from_jacobian":
                r.SetOptimizerScalesFromJacobian(
                    centralRegionRadius=self.central_region_radius
                )
            else:
                r.SetOptimizerScales(list(self.scales))
        if self.weights is not None:
            r.SetOptimizerWeights(list(self.weights))


class AmoebaSITKOptimizer(SITKOptimizer):
    simplex_delta: float
    num_iter: int
    params_convergence_tol: float = 1e-8
    func_convergence_tol: float = 1e-4
    with_restarts: bool = False

    def configure(self, r: sitk.ImageRegistrationMethod) -> None:
        super(AmoebaSITKOptimizer, self).configure(r)
        r.SetOptimizerAsAmoeba(
            self.simplex_delta,
            self.num_iter,
            parametersConvergenceTolerance=self.params_convergence_tol,
            functionConvergenceTolerance=self.func_convergence_tol,
            withRestarts=self.with_restarts,
        )


class ConjugateGradientLineSearchSITKOptimizer(SITKOptimizer):
    lr: float
    num_iter: int
    conv_min_val: float = 1e-6
    conv_window_size: int = 10
    line_search_lower: float = 0
    line_search_upper: float = 5
    line_search_eps: float = 0.01
    line_search_num_iter: int = 20
    lr_estim_type: str = "Once"
    max_step_size: float = 0

    def configure(self, r: sitk.ImageRegistrationMethod) -> None:
        super(ConjugateGradientLineSearchSITKOptimizer, self).configure(r)
        r.SetOptimizerAsConjugateGradientLineSearch(
            self.lr,
            self.num_iter,
            convergenceMinimumValue=self.conv_min_val,
            convergenceWindowSize=self.conv_window_size,
            lineSearchLowerLimit=self.line_search_lower,
            lineSearchUpperLimit=self.line_search_upper,
            lineSearchEpsilon=self.line_search_eps,
            lineSearchMaximumIterations=self.line_search_num_iter,
            estimateLearningRate=SITKOptimizer.LearningRateEstimationType[
                self.lr_estim_type
            ].value,
            maximumStepSizeInPhysicalUnits=self.max_step_size,
        )


class ExhaustiveSITKOptimizer(SITKOptimizer):
    num_steps: Sequence[int]
    step_length: float = 1

    def configure(self, r: sitk.ImageRegistrationMethod) -> None:
        super(ExhaustiveSITKOptimizer, self).configure(r)
        r.SetOptimizerAsExhaustive(list(self.num_steps), stepLength=self.step_length)


class GradientDescentSITKOptimizer(SITKOptimizer):
    lr: float
    num_iter: int
    conv_min_val: float = 1e-6
    conv_window_size: int = 10
    lr_estim_type: str = "Once"
    max_step_size: float = 0

    def configure(self, r: sitk.ImageRegistrationMethod) -> None:
        super(GradientDescentSITKOptimizer, self).configure(r)
        r.SetOptimizerAsGradientDescent(
            self.lr,
            self.num_iter,
            convergenceMinimumValue=self.conv_min_val,
            convergenceWindowSize=self.conv_window_size,
            estimateLearningRate=SITKOptimizer.LearningRateEstimationType[
                self.lr_estim_type
            ].value,
            maximumStepSizeInPhysicalUnits=self.max_step_size,
        )


class GradientDescentLineSearchSITKOptimizer(SITKOptimizer):
    lr: float
    num_iter: int
    conv_min_val: float = 1e-6
    conv_window_size: int = 10
    line_search_lower: float = 0
    line_search_upper: float = 5
    line_search_eps: float = 0.01
    line_search_num_iter: int = 20
    lr_estim_type: str = "Once"
    max_step_size: float = 0

    def configure(self, r: sitk.ImageRegistrationMethod) -> None:
        super(GradientDescentLineSearchSITKOptimizer, self).configure(r)
        r.SetOptimizerAsGradientDescentLineSearch(
            self.lr,
            self.num_iter,
            convergenceMinimumValue=self.conv_min_val,
            convergenceWindowSize=self.conv_window_size,
            lineSearchLowerLimit=self.line_search_lower,
            lineSearchUpperLimit=self.line_search_upper,
            lineSearchEpsilon=self.line_search_eps,
            lineSearchMaximumIterations=self.line_search_num_iter,
            estimateLearningRate=SITKOptimizer.LearningRateEstimationType[
                self.lr_estim_type
            ].value,
            maximumStepSizeInPhysicalUnits=self.max_step_size,
        )


class LBFGS2SITKOptimizer(SITKOptimizer):
    solution_accuracy: float = 1e-5
    num_iter: int = 0
    hessian_approx_accuracy: int = 6
    delta_conv_dist: int = 0
    delta_conv_tol: float = 1e-5
    line_search_max_eval: int = 40
    line_search_min_step: float = 1e-20
    line_search_max_step: float = 1e20
    line_search_accuracy: float = 1e-4

    def configure(self, r: sitk.ImageRegistrationMethod) -> None:
        super(LBFGS2SITKOptimizer, self).configure(r)
        r.SetOptimizerAsLBFGS2(
            solutionAccuracy=self.solution_accuracy,
            numberOfIterations=self.num_iter,
            hessianApproximateAccuracy=self.hessian_approx_accuracy,
            deltaConvergenceDistance=self.delta_conv_dist,
            deltaConvergenceTolerance=self.delta_conv_tol,
            lineSearchMaximumEvaluations=self.line_search_max_eval,
            lineSearchMinimumStep=self.line_search_min_step,
            lineSearchMaximumStep=self.line_search_max_step,
            lineSearchAccuracy=self.line_search_accuracy,
        )


class LBFGSBSITKOptimizer(SITKOptimizer):
    grad_conv_tol: float = 1e-5
    num_iter: int = 500
    max_num_corrs: int = 5
    max_num_func_evals: int = 2000
    cost_func_conv_factor: float = 1e7
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    trace: bool = False

    def configure(self, r: sitk.ImageRegistrationMethod) -> None:
        super(LBFGSBSITKOptimizer, self).configure(r)
        kwargs = {
            "gradientConvergenceTolerance": self.grad_conv_tol,
            "numberOfIterations": self.num_iter,
            "maximumNumberOfCorrections": self.max_num_corrs,
            "maximumNumberOfFunctionEvaluations": self.max_num_func_evals,
            "costFunctionConvergenceFactor": self.cost_func_conv_factor,
            "trace": self.trace,
        }
        if self.lower_bound is not None:
            kwargs["lowerBound"] = self.lower_bound
        if self.upper_bound is not None:
            kwargs["upperBound"] = self.upper_bound
        r.SetOptimizerAsLBFGSB(**kwargs)


class OnePlusOneEvolutionarySITKOptimizer(SITKOptimizer):
    num_iter: int = 100
    eps: float = 1.5e-4
    initial_radius: float = 1.01
    growth_factor: float = -1
    shrink_factor: float = -1
    seed: int = sitk.sitkWallClock

    def configure(self, r: sitk.ImageRegistrationMethod) -> None:
        super(OnePlusOneEvolutionarySITKOptimizer, self).configure(r)
        r.SetOptimizerAsOnePlusOneEvolutionary(
            numberOfIterations=self.num_iter,
            epsilon=self.eps,
            initialRadius=self.initial_radius,
            growthFactor=self.growth_factor,
            shrinkFactor=self.shrink_factor,
            seed=self.seed,
        )


class PowellSITKOptimizer(SITKOptimizer):
    num_iter: int = 100
    max_line_iter: int = 100
    step_length: float = 1
    step_tol: float = 1e-6
    val_tol: float = 1e-6

    def configure(self, r: sitk.ImageRegistrationMethod) -> None:
        super(PowellSITKOptimizer, self).configure(r)
        r.SetOptimizerAsPowell(
            numberOfIterations=self.num_iter,
            maximumLineIterations=self.max_line_iter,
            stepLength=self.step_length,
            stepTolerance=self.step_tol,
            valueTolerance=self.val_tol,
        )


class RegularStepGradientDescentSITKOptimizer(SITKOptimizer):
    lr: float
    min_step: float
    num_iter: int
    relax_factor: float = 0.5
    grad_magnitude_tol: float = 1e-4
    lr_estim_type: str = "Never"
    max_step_size: float = 0

    def configure(self, r: sitk.ImageRegistrationMethod) -> None:
        super(RegularStepGradientDescentSITKOptimizer, self).configure(r)
        r.SetOptimizerAsRegularStepGradientDescent(
            self.lr,
            self.min_step,
            self.num_iter,
            relaxationFactor=self.relax_factor,
            gradientMagnitudeTolerance=self.grad_magnitude_tol,
            estimateLearningRate=SITKOptimizer.LearningRateEstimationType[
                self.lr_estim_type
            ].value,
            maximumStepSizeInPhysicalUnits=self.max_step_size,
        )


sitk_optimizer_types: dict[str, type[SITKOptimizer]] = {
    "amoeba": AmoebaSITKOptimizer,
    "conjugate_gradient_line_search": ConjugateGradientLineSearchSITKOptimizer,
    "exhaustive": ExhaustiveSITKOptimizer,
    "gradient_descent": GradientDescentSITKOptimizer,
    "gradient_descent_line_search": GradientDescentLineSearchSITKOptimizer,
    "lbfgs2": LBFGS2SITKOptimizer,
    "lbfgsb": LBFGSBSITKOptimizer,
    "one_plus_one_evolutionary": OnePlusOneEvolutionarySITKOptimizer,
    "powell": PowellSITKOptimizer,
    "regular_step_gradient_descent": RegularStepGradientDescentSITKOptimizer,
}
