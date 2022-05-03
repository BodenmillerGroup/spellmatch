import math
import sys
from functools import cached_property
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Generator, Mapping, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from pydantic import BaseModel
from simutome import Simutome
from sklearn.model_selection import ParameterGrid

from ..io import read_scores
from ._benchmark import RunConfig, run_parallel, run_sequential


class SemisyntheticBenchmarkConfig(BaseModel):
    class AlgorithmConfig(BaseModel):
        algorithm_name: str
        algorithm_kwargs: dict[str, Any]
        algorithm_param_grid: dict[str, list[dict[str, Any]]]
        algorithm_is_directed: bool = False

    points_file_names: list[str]
    intensities_file_names: Optional[list[str]]
    clusters_file_names: Optional[list[str]]
    simutome_kwargs: dict[str, Any]
    simutome_param_grid: dict[str, list[dict[str, Any]]]
    n_simutome_sections: int
    algorithm_configs: dict[str, AlgorithmConfig]
    seed: int


class SemisyntheticBenchmark:
    AssignmentFunction = Callable[[xr.DataArray], xr.DataArray]  # FIXME reverse_scores
    MetricFunction = Callable[[np.ndarray, np.ndarray, np.ndarray], float]

    def __init__(
        self, benchmark_dir: Union[str, PathLike], config: SemisyntheticBenchmarkConfig
    ) -> None:
        self.benchmark_dir = Path(benchmark_dir)
        self.benchmark_dir.mkdir(exist_ok=True, parents=True)
        self.config = config

    def get_run_length(self, n_batches: int) -> int:
        return math.ceil(self.n_runs / n_batches)

    def run_sequential(
        self,
        points_dir: Union[str, PathLike],
        intensities_dir: Union[str, PathLike, None] = None,
        clusters_dir: Union[str, PathLike, None] = None,
        batch_index: int = 0,
        n_batches: int = 1,
    ) -> Generator[
        tuple[dict[str, Any], Optional[xr.DataArray], Optional[xr.DataArray]],
        None,
        pd.DataFrame,
    ]:
        run_config_generator = self.generate_run_configs(
            points_dir,
            intensities_dir=intensities_dir,
            clusters_dir=clusters_dir,
            batch_index=batch_index,
            n_batches=n_batches,
        )
        self.scores_info: pd.DataFrame = yield from run_sequential(
            run_config_generator,
            self.scores_dir,
            offset=batch_index * self.get_run_length(n_batches),
        )
        self.scores_info.to_csv(self.scores_info_file, index=False)
        return self.scores_info

    def run_parallel(
        self,
        points_dir: Union[str, PathLike],
        intensities_dir: Union[str, PathLike, None] = None,
        clusters_dir: Union[str, PathLike, None] = None,
        batch_index: int = 0,
        n_batches: int = 1,
        n_processes: Optional[int] = None,
        queue_size: int = None,
        worker_timeout: int = 1,
    ) -> Generator[RunConfig, None, pd.DataFrame]:
        run_config_generator = self.generate_run_configs(
            points_dir,
            intensities_dir=intensities_dir,
            clusters_dir=clusters_dir,
            batch_index=batch_index,
            n_batches=n_batches,
        )
        self.scores_info: pd.DataFrame = yield from run_parallel(
            run_config_generator,
            self.scores_dir,
            offset=batch_index * self.get_run_length(n_batches),
            n_processes=n_processes,
            queue_size=queue_size,
            worker_timeout=worker_timeout,
        )
        self.scores_info.to_csv(self.scores_info_file, index=False)
        return self.scores_info

    def generate_run_configs(
        self,
        points_dir: Union[str, PathLike],
        intensities_dir: Union[str, PathLike, None] = None,
        clusters_dir: Union[str, PathLike, None] = None,
        batch_index: int = 0,
        n_batches: int = 1,
    ) -> Generator[RunConfig, None, None]:
        points_dir = Path(points_dir)
        if intensities_dir is not None:
            intensities_dir = Path(intensities_dir)
        if clusters_dir is not None:
            clusters_dir = Path(clusters_dir)
        simutome_seeds = np.random.default_rng(seed=self.config.seed).integers(
            sys.maxsize, size=self.n_file_sets * self.n_simutome_params
        )
        n_runs_per_batch = self.get_run_length(n_batches)
        for run_config in self._generate_run_configs_for_batch(
            points_dir,
            intensities_dir,
            clusters_dir,
            simutome_seeds,
            batch_index * n_runs_per_batch,
            min((batch_index + 1) * n_runs_per_batch, self.n_runs),
        ):
            yield RunConfig(
                info={"batch": batch_index, **run_config.info},
                algorithm_name=run_config.algorithm_name,
                algorithm_kwargs=run_config.algorithm_kwargs,
                match_points_kwargs=run_config.match_points_kwargs,
                algorithm_is_directed=run_config.algorithm_is_directed,
            )

    def get_evaluation_length(
        self,
        assignment_functions: Mapping[str, AssignmentFunction],
        metric_functions: Optional[Mapping[str, MetricFunction]],
    ) -> int:
        return (
            np.sum(self.scores_info["scores_file"].notna())
            * len(assignment_functions)
            * len(metric_functions)
        )

    def evaluate(
        self,
        assignment_functions: Mapping[str, AssignmentFunction],
        metric_functions: Mapping[str, MetricFunction],
    ) -> Generator[dict[str, Any], None, pd.DataFrame]:
        results = []
        scores_info = self.scores_info.loc[self.scores_info["scores_file"].notna(), :]
        for _, scores_info_row in scores_info.iterrows():
            scores = read_scores(
                self.benchmark_dir / "scores" / scores_info_row["scores_file"]
            )
            scores_arr = scores.to_numpy()
            coords0 = scores.coords[scores.dims[0]].to_numpy()  # may contain duplicates
            coords1 = scores.coords[scores.dims[1]].to_numpy()  # may contain duplicates
            assignment_arr_true = np.zeros_like(scores_arr, dtype=bool)
            for c in np.unique(np.concatenate((coords0, coords1))):
                assignment_arr_true[np.ix_(coords0 == c, coords1 == c)] = True
            reverse_scores = None
            reverse_scores_arr = None
            reverse_assignment_arr_true = None
            reverse_scores_file = scores_info_row.get("reverse_scores_file")
            if reverse_scores_file not in (None, np.nan):
                reverse_scores = read_scores(
                    self.benchmark_dir / "scores" / reverse_scores_file
                )
                reverse_scores_arr = reverse_scores.to_numpy()
                reverse_coords0 = reverse_scores.coords[
                    reverse_scores.dims[0]
                ].to_numpy()  # may contain duplicates
                reverse_coords1 = reverse_scores.coords[
                    reverse_scores.dims[1]
                ].to_numpy()  # may contain duplicates
                reverse_assignment_arr_true = np.zeros_like(
                    reverse_scores_arr, dtype=bool
                )
                for c in np.unique(np.concatenate((reverse_coords0, reverse_coords1))):
                    reverse_assignment_arr_true[
                        np.ix_(reverse_coords0 == c, reverse_coords1 == c)
                    ] = True
            for assignment_name, assignment_function in assignment_functions.items():
                assignment_mat_pred: xr.DataArray = assignment_function(
                    scores, reverse_scores=reverse_scores
                )
                assignment_arr_pred = assignment_mat_pred.to_numpy()
                for metric_name, metric_function in metric_functions.items():
                    metric_value = metric_function(
                        scores_arr, assignment_arr_pred, assignment_arr_true
                    )
                    if (
                        reverse_scores_arr is not None
                        and reverse_assignment_arr_true is not None
                    ):
                        reverse_metric_value = metric_function(
                            reverse_scores_arr,
                            assignment_arr_pred,
                            reverse_assignment_arr_true,
                        )
                        metric_value = (metric_value + reverse_metric_value) / 2
                    result = {
                        **scores_info_row.to_dict(),
                        "assignment_name": assignment_name,
                        "metric_name": metric_name,
                        "metric_value": metric_value,
                    }
                    results.append(result)
                    yield result
        self.results_info = pd.DataFrame(data=results)
        self.results_info.to_csv(self.results_info_file, index=False)
        return self.results_info

    def save(self) -> None:
        config_dict = self.config.dict()
        with self.config_file.open("w") as f:
            yaml.dump(config_dict, f, sort_keys=False)

    @staticmethod
    def load(benchmark_dir: Union[str, PathLike]) -> "SemisyntheticBenchmark":
        config_file = Path(benchmark_dir) / "config.yml"
        with config_file.open("r") as f:
            config_dict = yaml.load(f, yaml.SafeLoader())
        config = SemisyntheticBenchmarkConfig.parse_obj(config_dict)
        return SemisyntheticBenchmark(config, benchmark_dir)

    def _generate_run_configs_for_batch(
        self,
        points_dir: Path,
        intensities_dir: Optional[Path],
        clusters_dir: Optional[Path],
        simutome_seeds: np.ndarray,
        start: int,
        stop: int,
    ) -> Generator[RunConfig, None, None]:
        offset = 0
        file_set_start = math.floor((start - offset) / self.n_runs_per_file_set)
        file_set_stop = math.ceil((stop - offset) / self.n_runs_per_file_set)
        assert 0 <= file_set_start < file_set_stop <= self.n_file_sets
        for file_set_index in range(file_set_start, file_set_stop):
            points_file_name = self.config.points_file_names[file_set_index]
            points = pd.read_csv(points_dir / points_file_name, index_col="cell")
            intensities_file_name = None
            intensities = None
            if (
                intensities_dir is not None
                and self.config.intensities_file_names is not None
            ):
                intensities_file_name = self.config.intensities_file_names[
                    file_set_index
                ]
                intensities = pd.read_csv(
                    intensities_dir / intensities_file_name, index_col="cell"
                )
            clusters_file_name = None
            clusters = None
            if clusters_dir is not None and self.config.clusters_file_names is not None:
                clusters_file_name = self.config.clusters_file_names[file_set_index]
                clusters = pd.read_csv(
                    clusters_dir / clusters_file_name, index_col="cell"
                ).iloc[:, 0]
            for run_config in self._generate_run_configs_for_file_set(
                file_set_index,
                points,
                intensities,
                clusters,
                simutome_seeds,
                max(start, offset + file_set_index * self.n_runs_per_file_set),
                min(stop, offset + (file_set_index + 1) * self.n_runs_per_file_set),
            ):
                yield RunConfig(
                    info={
                        "points_file_name": points_file_name,
                        "intensities_file_name": intensities_file_name,
                        "clusters_file_name": clusters_file_name,
                        **run_config.info,
                    },
                    algorithm_name=run_config.algorithm_name,
                    algorithm_kwargs=run_config.algorithm_kwargs,
                    match_points_kwargs=run_config.match_points_kwargs,
                )

    def _generate_run_configs_for_file_set(
        self,
        file_set_index: int,
        points: pd.DataFrame,
        intensities: Optional[pd.DataFrame],
        clusters: Optional[pd.Series],
        simutome_seeds: np.ndarray,
        start: int,
        stop: int,
    ) -> Generator[RunConfig, None, None]:
        offset = file_set_index * self.n_runs_per_file_set
        simutome_params_start = math.floor(
            (start - offset) / self.n_runs_per_simutome_params
        )
        simutome_params_stop = math.ceil(
            (stop - offset) / self.n_runs_per_simutome_params
        )
        assert (
            0 <= simutome_params_start < simutome_params_stop <= self.n_simutome_params
        )
        simutome_param_grid = ParameterGrid(self.config.simutome_param_grid)
        for simutome_params_index in range(simutome_params_start, simutome_params_stop):
            simutome_params: dict[str, dict[str, Any]] = simutome_param_grid[
                simutome_params_index
            ]
            varying_simutome_kwargs = {
                k: v
                for simutome_param_group in simutome_params.values()
                for k, v in simutome_param_group.items()
            }
            simutome = Simutome(
                **self.config.simutome_kwargs,
                **varying_simutome_kwargs,
                seed=simutome_seeds[
                    file_set_index * self.n_simutome_params + simutome_params_index
                ],
            )
            for run_config in self._generate_run_configs_for_simutome_params(
                file_set_index,
                simutome_params_index,
                points,
                intensities,
                clusters,
                simutome,
                max(
                    start,
                    offset + simutome_params_index * self.n_runs_per_simutome_params,
                ),
                min(
                    stop,
                    offset
                    + (simutome_params_index + 1) * self.n_runs_per_simutome_params,
                ),
            ):
                yield RunConfig(
                    info={
                        **{
                            f"simutome_{k}": v
                            for k, v in varying_simutome_kwargs.items()
                        },
                        **run_config.info,
                    },
                    algorithm_name=run_config.algorithm_name,
                    algorithm_kwargs=run_config.algorithm_kwargs,
                    match_points_kwargs=run_config.match_points_kwargs,
                )

    def _generate_run_configs_for_simutome_params(
        self,
        file_set_index: int,
        simutome_params_index: int,
        points: pd.DataFrame,
        intensities: Optional[pd.DataFrame],
        clusters: Optional[pd.Series],
        simutome: Simutome,
        start: int,
        stop: int,
    ) -> Generator[RunConfig, None, None]:
        offset = (
            file_set_index * self.n_runs_per_file_set
            + simutome_params_index * self.n_runs_per_simutome_params
        )
        simutome_sections_start = math.floor(
            (start - offset) / self.n_runs_per_simutome_sections
        )
        simutome_sections_stop = math.ceil(
            (stop - offset) / self.n_runs_per_simutome_sections
        )
        assert (
            0
            <= simutome_sections_start
            < simutome_sections_stop
            <= self.n_simutome_sections
        )
        cell_points = points.to_numpy()
        cell_intensities = None
        if intensities is not None:
            cell_intensities = intensities.loc[points.index, :].to_numpy()
        cell_clusters = None
        if clusters is not None:
            cell_clusters = clusters.loc[points.index].to_numpy()
        simutome.skip_section_pairs(simutome_sections_start)
        section_pair_generator = simutome.generate_section_pairs(
            cell_points,
            cell_intensities=cell_intensities,
            cell_clusters=cell_clusters,
            n=simutome_sections_stop - simutome_sections_start,
        )
        for simutome_sections_index, (
            (source_indices, source_new_mask, source_points, source_intensities),
            (target_indices, target_new_mask, target_points, target_intensities),
        ) in enumerate(section_pair_generator, start=simutome_sections_start):
            next_new_cell_id = np.amax(points.index) + 1
            source_n_new = np.sum(source_new_mask)
            source_index = points.index[source_indices].to_numpy().copy()
            source_index[source_new_mask] = next_new_cell_id + np.arange(source_n_new)
            source_index = pd.Index(source_index, name="cell")
            source_points = pd.DataFrame(
                source_points, index=source_index, columns=points.columns
            )
            if source_intensities is not None:
                source_intensities = pd.DataFrame(
                    source_intensities, index=source_index, columns=intensities.columns
                )
            next_new_cell_id += source_n_new
            target_n_new = np.sum(target_new_mask)
            target_index = points.index[target_indices].to_numpy().copy()
            target_index[target_new_mask] = next_new_cell_id + np.arange(target_n_new)
            target_index = pd.Index(target_index, name="cell")
            target_points = pd.DataFrame(
                target_points, index=target_index, columns=points.columns
            )
            if target_intensities is not None:
                target_intensities = pd.DataFrame(
                    target_intensities, index=target_index, columns=intensities.columns
                )
            next_new_cell_id += target_n_new
            for run_config in self._generate_run_configs_for_simutome_sections(
                file_set_index,
                simutome_params_index,
                simutome_sections_index,
                max(
                    start,
                    offset
                    + simutome_sections_index * self.n_runs_per_simutome_sections,
                ),
                min(
                    stop,
                    offset
                    + (simutome_sections_index + 1) * self.n_runs_per_simutome_sections,
                ),
            ):
                yield RunConfig(
                    info={"section_number": simutome_sections_index, **run_config.info},
                    algorithm_name=run_config.algorithm_name,
                    algorithm_kwargs=run_config.algorithm_kwargs,
                    match_points_kwargs={
                        "source_name": "source",
                        "source_points": source_points,
                        "source_intensities": source_intensities,
                        "target_name": "target",
                        "target_points": target_points,
                        "target_intensities": target_intensities,
                        **run_config.match_points_kwargs,
                    },
                )

    def _generate_run_configs_for_simutome_sections(
        self,
        file_set_index: int,
        simutome_params_index: int,
        simutome_sections_index: int,
        start: int,
        stop: int,
    ) -> Generator[RunConfig, None, None]:
        offset = (
            file_set_index * self.n_runs_per_file_set
            + simutome_params_index * self.n_runs_per_simutome_params
            + simutome_sections_index * self.n_runs_per_simutome_sections
        )
        i = 0
        for (
            algorithm_config_name,
            algorithm_config,
        ) in self.config.algorithm_configs.items():
            algorithm_param_grid = ParameterGrid(algorithm_config.algorithm_param_grid)
            for algorithm_params in algorithm_param_grid:
                algorithm_params: dict[str, dict[str, Any]]
                if start <= offset + i < stop:
                    varying_algorithm_kwargs = {
                        k: v
                        for algorithm_param_group in algorithm_params.values()
                        for k, v in algorithm_param_group.items()
                    }
                    yield RunConfig(
                        info={
                            "algorithm_config_name": algorithm_config_name,
                            **{
                                f"{algorithm_config.algorithm_name}_{k}": v
                                for k, v in varying_algorithm_kwargs.items()
                            },
                        },
                        algorithm_name=algorithm_config.algorithm_name,
                        algorithm_kwargs={
                            **algorithm_config.algorithm_kwargs,
                            **varying_algorithm_kwargs,
                        },
                        match_points_kwargs={},
                    )
                i += 1

    @property
    def config_file(self) -> Path:
        return self.benchmark_dir / "config.yaml"

    @property
    def scores_dir(self) -> Path:
        return self.benchmark_dir / "scores"

    @property
    def scores_info_file(self) -> Path:
        return self.benchmark_dir / "scores.csv"

    @cached_property
    def scores_info(self) -> pd.DataFrame:
        return pd.read_csv(self.scores_info_file)

    @property
    def results_info_file(self) -> Path:
        return self.benchmark_dir / "results.csv"

    @cached_property
    def results_info(self) -> pd.DataFrame:
        return pd.read_csv(self.results_info_file)

    @property
    def n_file_sets(self) -> int:
        return len(self.config.points_file_names)

    @property
    def n_simutome_params(self) -> int:
        return len(ParameterGrid(self.config.simutome_param_grid))

    @property
    def n_simutome_sections(self) -> int:
        return self.config.n_simutome_sections

    @property
    def n_algorithm_configs_param_grids(self) -> int:
        return sum(
            len(ParameterGrid(algorithm_config.algorithm_param_grid))
            for algorithm_config in self.config.algorithm_configs.values()
        )

    @property
    def n_runs(self) -> int:
        return (
            self.n_file_sets
            * self.n_simutome_params
            * self.n_simutome_sections
            * self.n_algorithm_configs_param_grids
        )

    @property
    def n_runs_per_file_set(self) -> int:
        return (
            self.n_simutome_params
            * self.n_simutome_sections
            * self.n_algorithm_configs_param_grids
        )

    @property
    def n_runs_per_simutome_params(self) -> int:
        return self.n_simutome_sections * self.n_algorithm_configs_param_grids

    @property
    def n_runs_per_simutome_sections(self) -> int:
        return self.n_algorithm_configs_param_grids
