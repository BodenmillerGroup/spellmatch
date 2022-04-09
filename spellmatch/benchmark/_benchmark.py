from os import PathLike
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Callable, Generator, Optional, Sequence, Tuple, Type, Union

import numpy as np
import pandas as pd
import xarray as xr
from simutome import Simutome
from sklearn.model_selection import ParameterGrid

from ..matching.algorithms import PointsMatchingAlgorithm

AlgorithmConfig = Tuple[
    Type[PointsMatchingAlgorithm],
    Optional[dict[str, Any]],
    Optional[ParameterGrid],
    dict[str, Callable[[xr.DataArray], xr.DataArray]],
]


def run_benchmark(
    source_points_files: Sequence[Union[str, PathLike]],
    source_intensities_files: Optional[Sequence[Union[str, PathLike]]],
    source_clusters_files: Optional[Sequence[Union[str, PathLike]]],
    simutome_kwargs: Optional[dict[str, Any]],
    simutome_param_grid: Optional[ParameterGrid],
    num_sections: int,
    algorithm_dict: dict[str, AlgorithmConfig],
    metric_dict: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], float]],
    seed=None,
) -> Generator[Tuple[dict[str, Any], xr.DataArray, list[dict[str, Any]]], None, None]:
    for source_points_file, source_intensities_file, source_clusters_file in zip(
        source_points_files,
        (source_intensities_files or [None] * len(source_points_files)),
        (source_clusters_files or [None] * len(source_points_files)),
    ):
        source_points = pd.read_csv(source_points_file, index_col="cell")
        source_intensities = None
        if source_intensities_file is not None:
            source_intensities = pd.read_csv(source_intensities_file, index_col="cell")
        source_clusters = None
        if source_clusters_file is not None:
            source_clusters = pd.read_csv(source_clusters_file, index_col="cell")
            source_clusters = source_clusters.iloc[:, 0]
        for info, scores, results in _evaluate_simutome(
            source_points,
            source_intensities,
            source_clusters,
            simutome_kwargs,
            simutome_param_grid,
            num_sections,
            algorithm_dict,
            metric_dict,
            seed,
        ):
            info = {
                "source_points_file": Path(source_points_file).name,
                "source_intensities_file": (
                    Path(source_intensities_file).name
                    if source_intensities_file is not None
                    else np.nan
                ),
                "source_clusters_file": (
                    Path(source_clusters_file).name
                    if source_clusters_file is not None
                    else np.nan
                ),
                **info,
            }
            yield info, scores, results


def _evaluate_simutome(
    source_points: pd.DataFrame,
    source_intensities: Optional[pd.DataFrame],
    source_clusters: Optional[pd.Series],
    simutome_kwargs: Optional[dict[str, Any]],
    simutome_param_grid: Optional[ParameterGrid],
    num_sections: int,
    algorithm_dict: dict[str, AlgorithmConfig],
    metric_dict: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], float]],
    seed,
) -> Generator[Tuple[dict[str, Any], xr.DataArray, list[dict[str, Any]]], None, None]:
    for simutome_params in simutome_param_grid or [{}]:
        simutome_params = {k: v for p in simutome_params.values() for k, v in p.items()}
        cur_simutome_kwargs = {}
        if simutome_kwargs is not None:
            cur_simutome_kwargs.update(simutome_kwargs)
        cur_simutome_kwargs.update(simutome_params)
        simutome = Simutome(**cur_simutome_kwargs, seed=seed)
        section_generator = simutome.generate_sections(
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
            n=num_sections,
        )
        for (
            section_number,
            (source_indices, target_points, target_intensities),
        ) in enumerate(section_generator):
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
            for info, scores, results in _evaluate_algorithms(
                source_points,
                target_points,
                source_intensities,
                target_intensities,
                algorithm_dict,
                metric_dict,
                assignment_true,
            ):
                info = {
                    **{f"simutome_{k}": v for k, v in simutome_params.items()},
                    "section_number": section_number,
                    **info,
                }
                yield info, scores, results


def _evaluate_algorithms(
    source_points: pd.DataFrame,
    target_points: pd.DataFrame,
    source_intensities: Optional[pd.DataFrame],
    target_intensities: Optional[pd.DataFrame],
    algorithm_dict: dict[str, AlgorithmConfig],
    metric_dict: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], float]],
    assignment_true: xr.DataArray,
) -> Generator[Tuple[dict[str, Any], xr.DataArray, list[dict[str, Any]]], None, None]:
    for (
        algorithm_name,
        (algorithm_type, algorithm_kwargs, algorithm_param_grid, assignment_dict),
    ) in algorithm_dict.items():
        for algorithm_params in algorithm_param_grid or [{}]:
            algorithm_params = {
                k: v for p in algorithm_params.values() for k, v in p.items()
            }
            cur_algorithm_kwargs = {}
            if algorithm_kwargs is not None:
                cur_algorithm_kwargs.update(algorithm_kwargs)
            cur_algorithm_kwargs.update(algorithm_params)
            algorithm = algorithm_type(**cur_algorithm_kwargs)
            start = timer()
            scores = algorithm.match_points(
                "source",
                "simutome",
                source_points,
                target_points,
                source_intensities=source_intensities,
                target_intensities=target_intensities,
            )
            end = timer()
            results = _evaluate_scores(
                scores, assignment_dict, metric_dict, assignment_true
            )
            info = {
                "algorithm_name": algorithm_name,
                **{f"{algorithm_name}_{k}": v for k, v in algorithm_params.items()},
                "seconds": end - start,
            }
            yield info, scores, results


def _evaluate_scores(
    scores: xr.DataArray,
    assignment_dict: dict[str, Callable[[xr.DataArray], xr.DataArray]],
    metric_dict: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], float]],
    assignment_true: xr.DataArray,
) -> list[dict[str, Any]]:
    results = []
    scores_arr = scores.to_numpy()
    assignment_arr_true = assignment_true.loc[scores.coords].to_numpy()
    for assignment_name, assignment_fn in assignment_dict.items():
        assignment_arr_pred = assignment_fn(scores).loc[scores.coords].to_numpy()
        for metric_name, metric_fn in metric_dict.items():
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
