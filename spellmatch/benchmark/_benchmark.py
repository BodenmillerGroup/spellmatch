from os import PathLike
from pathlib import Path
from timeit import default_timer as timer
from typing import (
    Any,
    Callable,
    Generator,
    Iterator,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import numpy as np
import pandas as pd
import xarray as xr
from simutome import Simutome
from sklearn.model_selection import ParameterGrid

from ..matching.algorithms import (
    PointsMatchingAlgorithm,
    SpellmatchMatchingAlgorithmException,
)

AssignmentFunction = Callable[[xr.DataArray], xr.DataArray]
MetricFunction = Callable[[np.ndarray, np.ndarray, np.ndarray], float]


class AlgorithmConfig(NamedTuple):
    algorithm_type: Type[PointsMatchingAlgorithm]
    algorithm_kwargs: Mapping[str, Any] = {}
    algorithm_param_grid: ParameterGrid = ParameterGrid({})
    assignment_functions: Mapping[str, AssignmentFunction] = {}


class Benchmark:
    def __init__(
        self,
        *,
        source_points_files: Sequence[Union[str, PathLike]],
        source_intensities_files: Optional[Sequence[Union[str, PathLike]]] = None,
        source_clusters_files: Optional[Sequence[Union[str, PathLike]]] = None,
        simutome_kwargs: Optional[Mapping[str, Any]] = None,
        simutome_param_grid: Optional[ParameterGrid] = None,
        num_sections: int = 1,
        algorithm_configs: Mapping[str, AlgorithmConfig] = None,
        metric_functions: Optional[Mapping[str, MetricFunction]] = None,
        seed=None,
    ) -> None:
        if len(source_points_files) == 0:
            raise ValueError("source_points_files")
        if source_intensities_files is not None and len(
            source_intensities_files
        ) != len(source_points_files):
            raise ValueError("source_intensities_files")
        if source_clusters_files is not None and len(source_clusters_files) != len(
            source_points_files
        ):
            raise ValueError("source_clusters_files")
        if num_sections <= 0:
            raise ValueError("num_sections")
        self.source_points_files = source_points_files
        self.source_intensities_files = source_intensities_files or [None] * len(
            source_points_files
        )
        self.source_clusters_files = source_clusters_files or [None] * len(
            source_points_files
        )
        self.simutome_kwargs = simutome_kwargs or {}
        self.simutome_param_grid = simutome_param_grid or ParameterGrid({})
        self.num_sections = num_sections
        self.algorithm_configs = algorithm_configs or {}
        self.metric_functions = metric_functions or {}
        self._rng = np.random.default_rng(seed=seed)

    def __len__(self) -> int:
        return (
            len(self.source_points_files)
            * len(self.simutome_param_grid)
            * self.num_sections
            * sum(
                len(algorithm_config.algorithm_param_grid)
                for algorithm_config in self.algorithm_configs.values()
            )
        )

    def __iter__(
        self,
    ) -> Iterator[
        Tuple[dict[str, Any], Optional[xr.DataArray], Optional[list[dict[str, Any]]]]
    ]:
        for source_points_file, source_intensities_file, source_clusters_file in zip(
            self.source_points_files,
            self.source_intensities_files,
            self.source_clusters_files,
        ):
            source_points = pd.read_csv(source_points_file, index_col="cell")
            source_intensities = None
            if source_intensities_file is not None:
                source_intensities = pd.read_csv(
                    source_intensities_file, index_col="cell"
                )
            source_clusters = None
            if source_clusters_file is not None:
                source_clusters = pd.read_csv(source_clusters_file, index_col="cell")
                source_clusters = source_clusters.iloc[:, 0]
            for info, scores, results in self._evaluate_simutome(
                source_points,
                source_intensities=source_intensities,
                source_clusters=source_clusters,
            ):
                info = {
                    "source_points_file": Path(source_points_file).name,
                    "source_intensities_file": (
                        Path(source_intensities_file).name
                        if source_intensities_file is not None
                        else None
                    ),
                    "source_clusters_file": (
                        Path(source_clusters_file).name
                        if source_clusters_file is not None
                        else None
                    ),
                    **info,
                }
                yield info, scores, results

    def _evaluate_simutome(
        self,
        source_points: pd.DataFrame,
        source_intensities: Optional[pd.DataFrame] = None,
        source_clusters: Optional[pd.Series] = None,
    ) -> Generator[
        Tuple[dict[str, Any], Optional[xr.DataArray], Optional[list[dict[str, Any]]]],
        None,
        None,
    ]:
        for simutome_params in self.simutome_param_grid:
            flat_simutome_params = {
                k: v
                for simutome_param_group in simutome_params.values()
                for k, v in simutome_param_group.items()
            }
            simutome = Simutome(
                **self.simutome_kwargs, **flat_simutome_params, seed=self._rng
            )
            section_gen = simutome.generate_sections(
                source_points.to_numpy(),
                cell_intensities=(
                    source_intensities.loc[source_points.index, :].to_numpy()
                    if source_intensities is not None
                    else None
                ),
                cell_clusters=(
                    source_clusters.loc[source_points.index].to_numpy()
                    if source_clusters is not None
                    else None
                ),
                n=self.num_sections,
            )
            for (
                section_number,
                (source_indices, target_points, target_intensities),
            ) in enumerate(section_gen):
                target_points = pd.DataFrame(
                    target_points,
                    index=source_points.index[source_indices],
                    columns=source_points.columns,
                )
                if target_intensities is not None:
                    target_intensities = pd.DataFrame(
                        target_intensities,
                        index=source_intensities.index[source_indices],
                        columns=source_intensities.columns,
                    )
                assignment_true_data = np.zeros(
                    (len(source_points.index), len(target_points.index)), dtype=bool
                )
                assignment_true_data[
                    source_indices, np.arange(assignment_true_data.shape[1])
                ] = True
                assignment_true = xr.DataArray(
                    data=assignment_true_data,
                    coords={
                        "source": source_points.index.to_numpy(),
                        "simutome": target_points.index.to_numpy(),
                    },
                )
                for info, scores, results in self._evaluate_algorithms(
                    source_points,
                    target_points,
                    assignment_true,
                    source_intensities=source_intensities,
                    target_intensities=target_intensities,
                ):
                    info = {
                        **{f"simutome_{k}": v for k, v in flat_simutome_params.items()},
                        "section_number": section_number,
                        **info,
                    }
                    yield info, scores, results

    def _evaluate_algorithms(
        self,
        source_points: pd.DataFrame,
        target_points: pd.DataFrame,
        assignment_true: xr.DataArray,
        source_intensities: Optional[pd.DataFrame] = None,
        target_intensities: Optional[pd.DataFrame] = None,
    ) -> Generator[
        Tuple[dict[str, Any], Optional[xr.DataArray], Optional[list[dict[str, Any]]]],
        None,
        None,
    ]:
        for algorithm_name, algorithm_config in self.algorithm_configs.items():
            for algorithm_params in algorithm_config.algorithm_param_grid:
                flat_algorithm_params = {
                    k: v
                    for algorithm_param_group in algorithm_params.values()
                    for k, v in algorithm_param_group.items()
                }
                algorithm = algorithm_config.algorithm_type(
                    **algorithm_config.algorithm_kwargs, **flat_algorithm_params
                )
                scores = None
                error = None
                start = timer()
                try:
                    scores = algorithm.match_points(
                        "source",
                        "simutome",
                        source_points,
                        target_points,
                        source_intensities=source_intensities,
                        target_intensities=target_intensities,
                    )
                except SpellmatchMatchingAlgorithmException as e:
                    error = str(e)
                end = timer()
                results = None
                if scores is not None:
                    results = self._evaluate_scores(
                        scores, assignment_true, algorithm_config.assignment_functions
                    )
                info = {
                    "algorithm_name": algorithm_name,
                    **{
                        f"{algorithm_name}_{k}": v
                        for k, v in flat_algorithm_params.items()
                    },
                    "seconds": end - start,
                    "success": scores is not None,
                    "error": error,
                }
                yield info, scores, results

    def _evaluate_scores(
        self,
        scores: xr.DataArray,
        assignment_true: xr.DataArray,
        assignment_functions: Mapping[str, AssignmentFunction],
    ) -> list[dict[str, Any]]:
        results = []
        scores_arr = scores.to_numpy()
        assignment_arr_true = assignment_true.loc[scores.coords].to_numpy()
        for assignment_name, assignment_fn in assignment_functions.items():
            assignment_arr_pred = assignment_fn(scores).loc[scores.coords].to_numpy()
            for metric_name, metric_fn in self.metric_functions.items():
                metric_value = metric_fn(
                    scores_arr, assignment_arr_pred, assignment_arr_true
                )
                results.append(
                    {
                        "assignment_name": assignment_name,
                        "metric_name": metric_name,
                        "metric_value": metric_value,
                    }
                )
        return results
