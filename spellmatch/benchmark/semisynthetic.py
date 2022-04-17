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

    source_points_file_names: list[str]
    source_intensities_file_names: Optional[list[str]]
    source_clusters_file_names: Optional[list[str]]
    simutome_kwargs: dict[str, Any]
    simutome_param_grid: dict[str, list[dict[str, Any]]]
    n_simutome_sections: int
    algorithm_configs: dict[str, AlgorithmConfig]
    seed: int


class SemisyntheticBenchmark:
    AssignmentFunction = Callable[[xr.DataArray], xr.DataArray]
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
        source_points_dir: Union[str, PathLike],
        source_intensities_dir: Union[str, PathLike, None] = None,
        source_clusters_dir: Union[str, PathLike, None] = None,
        batch_index: int = 0,
        n_batches: int = 1,
    ) -> Generator[tuple[dict[str, Any], xr.DataArray], None, pd.DataFrame]:
        run_config_generator = self.generate_run_configs(
            source_points_dir,
            source_intensities_dir=source_intensities_dir,
            source_clusters_dir=source_clusters_dir,
            batch_index=batch_index,
            n_batches=n_batches,
        )
        self.scores_info: pd.DataFrame = yield from run_sequential(
            run_config_generator, self.scores_dir
        )
        self.scores_info.to_csv(self.scores_info_file, index=False)
        return self.scores_info

    def run_parallel(
        self,
        source_points_dir: Union[str, PathLike],
        source_intensities_dir: Union[str, PathLike, None] = None,
        source_clusters_dir: Union[str, PathLike, None] = None,
        batch_index: int = 0,
        n_batches: int = 1,
        n_processes: Optional[int] = None,
        queue_size: int = None,
    ) -> Generator[RunConfig, None, pd.DataFrame]:
        run_config_generator = self.generate_run_configs(
            source_points_dir,
            source_intensities_dir=source_intensities_dir,
            source_clusters_dir=source_clusters_dir,
            batch_index=batch_index,
            n_batches=n_batches,
        )
        self.scores_info: pd.DataFrame = yield from run_parallel(
            run_config_generator,
            self.scores_dir,
            n_processes=n_processes,
            queue_size=queue_size,
        )
        self.scores_info.to_csv(self.scores_info_file, index=False)
        return self.scores_info

    def generate_run_configs(
        self,
        source_points_dir: Union[str, PathLike],
        source_intensities_dir: Union[str, PathLike, None] = None,
        source_clusters_dir: Union[str, PathLike, None] = None,
        batch_index: int = 0,
        n_batches: int = 1,
    ) -> Generator[RunConfig, None, None]:
        source_points_dir = Path(source_points_dir)
        if source_intensities_dir is not None:
            source_intensities_dir = Path(source_intensities_dir)
        if source_clusters_dir is not None:
            source_clusters_dir = Path(source_clusters_dir)
        simutome_seeds = np.random.default_rng(seed=self.config.seed).integers(
            sys.maxsize, size=self.n_file_sets * self.n_simutome_params
        )
        n_runs_per_batch = self.get_run_length(n_batches)
        for run_config in self._generate_run_configs_for_batch(
            source_points_dir,
            source_intensities_dir,
            source_clusters_dir,
            simutome_seeds,
            batch_index * n_runs_per_batch,
            min((batch_index + 1) * n_runs_per_batch, self.n_runs),
        ):
            yield RunConfig(
                info={"batch": batch_index, **run_config.info},
                algorithm_name=run_config.algorithm_name,
                algorithm_kwargs=run_config.algorithm_kwargs,
                match_points_kwargs=run_config.match_points_kwargs,
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
            assignment_mat_true = scores.copy(
                data=np.zeros_like(scores_arr, dtype=bool)
            )
            cell_identity_indexer = xr.DataArray(
                data=np.intersect1d(
                    scores.coords[scores.dims[0]].to_numpy(),
                    scores.coords[scores.dims[1]].to_numpy(),
                )
            )
            assignment_mat_true.loc[cell_identity_indexer, cell_identity_indexer] = True
            assignment_arr_true = assignment_mat_true.to_numpy()
            for assignment_name, assignment_function in assignment_functions.items():
                assignment_mat_pred = assignment_function(scores)
                assignment_arr_pred = assignment_mat_pred.loc[scores.coords].to_numpy()
                for metric_name, metric_function in metric_functions.items():
                    metric_value = metric_function(
                        scores_arr, assignment_arr_pred, assignment_arr_true
                    )
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
        source_points_dir: Path,
        source_intensities_dir: Optional[Path],
        source_clusters_dir: Optional[Path],
        simutome_seeds: np.ndarray,
        start: int,
        stop: int,
    ) -> Generator[RunConfig, None, None]:
        offset = 0
        file_set_start = math.floor((start - offset) / self.n_runs_per_file_set)
        file_set_stop = math.ceil((stop - offset) / self.n_runs_per_file_set)
        assert 0 <= file_set_start < file_set_stop <= self.n_file_sets
        for file_set_index in range(file_set_start, file_set_stop):
            source_points_file_name = self.config.source_points_file_names[
                file_set_index
            ]
            source_points = pd.read_csv(
                source_points_dir / source_points_file_name, index_col="cell"
            )
            source_intensities_file_name = None
            source_intensities = None
            if (
                source_intensities_dir is not None
                and self.config.source_intensities_file_names is not None
            ):
                source_intensities_file_name = (
                    self.config.source_intensities_file_names[file_set_index]
                )
                source_intensities = pd.read_csv(
                    source_intensities_dir / source_intensities_file_name,
                    index_col="cell",
                )
            source_clusters_file_name = None
            source_clusters = None
            if (
                source_clusters_dir is not None
                and self.config.source_clusters_file_names is not None
            ):
                source_clusters_file_name = self.config.source_clusters_file_names[
                    file_set_index
                ]
                source_clusters = pd.read_csv(
                    source_clusters_dir / source_clusters_file_name, index_col="cell"
                ).iloc[:, 0]
            for run_config in self._generate_run_configs_for_file_set(
                file_set_index,
                source_points,
                source_intensities,
                source_clusters,
                simutome_seeds,
                max(start, offset + file_set_index * self.n_runs_per_file_set),
                min(stop, offset + (file_set_index + 1) * self.n_runs_per_file_set),
            ):
                yield RunConfig(
                    info={
                        "source_points_file_name": source_points_file_name,
                        "source_intensities_file_name": source_intensities_file_name,
                        "source_clusters_file_name": source_clusters_file_name,
                        **run_config.info,
                    },
                    algorithm_name=run_config.algorithm_name,
                    algorithm_kwargs=run_config.algorithm_kwargs,
                    match_points_kwargs={
                        "source_name": "source",
                        "source_points": source_points,
                        "source_intensities": source_intensities,
                        **run_config.match_points_kwargs,
                    },
                )

    def _generate_run_configs_for_file_set(
        self,
        file_set_index: int,
        source_points: pd.DataFrame,
        source_intensities: Optional[pd.DataFrame],
        source_clusters: Optional[pd.Series],
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
                source_points,
                source_intensities,
                source_clusters,
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
        source_points: pd.DataFrame,
        source_intensities: Optional[pd.DataFrame],
        source_clusters: Optional[pd.Series],
        simutome: Simutome,
        start: int,
        stop: int,
    ) -> Generator[RunConfig, None, None]:
        offset = (
            file_set_index * self.n_runs_per_file_set
            + simutome_params_index * self.n_runs_per_simutome_params
        )
        simutome_section_start = math.floor(
            (start - offset) / self.n_runs_per_simutome_section
        )
        simutome_section_stop = math.ceil(
            (stop - offset) / self.n_runs_per_simutome_section
        )
        assert (
            0
            <= simutome_section_start
            < simutome_section_stop
            <= self.n_simutome_sections
        )
        simutome.skip_sections(simutome_section_start)
        simutome_section_generator = simutome.generate_sections(
            source_points.to_numpy(),
            cell_intensities=source_intensities.loc[source_points.index, :].to_numpy()
            if source_intensities is not None
            else None,
            cell_clusters=source_clusters.loc[source_points.index].to_numpy()
            if source_clusters is not None
            else None,
            n=simutome_section_stop - simutome_section_start,
        )
        for simutome_section_index, (
            source_indices,
            target_points,
            target_intensities,
        ) in enumerate(simutome_section_generator, start=simutome_section_start):
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
            for run_config in self._generate_run_configs_for_simutome_section(
                file_set_index,
                simutome_params_index,
                simutome_section_index,
                max(
                    start,
                    offset + simutome_section_index * self.n_runs_per_simutome_section,
                ),
                min(
                    stop,
                    offset
                    + (simutome_section_index + 1) * self.n_runs_per_simutome_section,
                ),
            ):
                yield RunConfig(
                    info={"section_number": simutome_section_index, **run_config.info},
                    algorithm_name=run_config.algorithm_name,
                    algorithm_kwargs=run_config.algorithm_kwargs,
                    match_points_kwargs={
                        "target_name": "simutome",
                        "target_points": target_points,
                        "target_intensities": target_intensities,
                        **run_config.match_points_kwargs,
                    },
                )

    def _generate_run_configs_for_simutome_section(
        self,
        file_set_index: int,
        simutome_params_index: int,
        simutome_section_index: int,
        start: int,
        stop: int,
    ) -> Generator[RunConfig, None, None]:
        offset = (
            file_set_index * self.n_runs_per_file_set
            + simutome_params_index * self.n_runs_per_simutome_params
            + simutome_section_index * self.n_runs_per_simutome_section
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
        return len(self.config.source_points_file_names)

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
    def n_runs_per_simutome_section(self) -> int:
        return self.n_algorithm_configs_param_grids
