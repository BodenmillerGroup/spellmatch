import multiprocessing as mp
import queue
from collections.abc import Generator, Iterable
from os import PathLike, cpu_count
from pathlib import Path
from timeit import default_timer as timer
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import BaseModel
from skimage.transform import ProjectiveTransform

from ..io import write_scores
from ..matching.algorithms import PointsMatchingAlgorithm
from ..plugins import plugin_manager


class RunConfig(BaseModel):
    info: dict[str, Any]
    algorithm_name: str
    algorithm_kwargs: dict[str, Any]
    match_points_kwargs: dict[str, Any]
    algorithm_is_directed: bool = False


def run(
    run_config: RunConfig,
    scores_file: str | PathLike,
    reverse_scores_file: str | PathLike | None = None,
) -> tuple[dict[str, Any], xr.DataArray | None, xr.DataArray | None]:
    scores_file = Path(scores_file)
    if reverse_scores_file is not None:
        reverse_scores_file = Path(reverse_scores_file)
    scores_info = run_config.info.copy()
    algorithm = None
    try:
        algorithm_types: list[type[PointsMatchingAlgorithm]] = (
            plugin_manager.hook.spellmatch_get_mask_matching_algorithm(
                name=run_config.algorithm_name
            )
        )
        assert len(algorithm_types) == 1
        algorithm_type = algorithm_types[0]
        algorithm = algorithm_type(**run_config.algorithm_kwargs)
    except Exception as e:
        scores_info["error"] = str(e)
    info = None
    scores = None
    reverse_info = None
    reverse_scores = None
    if algorithm is not None:
        try:
            start = timer()
            info, scores = algorithm.match_points(**run_config.match_points_kwargs)
            if run_config.algorithm_is_directed:
                inv_prior_transform = None
                if "prior_transform" in run_config.match_points_kwargs:
                    inv_prior_transform = ProjectiveTransform(
                        matrix=np.linalg.inv(
                            run_config.match_points_kwargs["prior_transform"].params
                        )
                    )
                reverse_info, reverse_scores = algorithm.match_points(
                    run_config.match_points_kwargs["target_name"],
                    run_config.match_points_kwargs["source_name"],
                    run_config.match_points_kwargs["target_points"],
                    run_config.match_points_kwargs["source_points"],
                    source_bbox=run_config.match_points_kwargs.get("target_bbox"),
                    target_bbox=run_config.match_points_kwargs.get("source_bbox"),
                    source_intensities=run_config.match_points_kwargs.get(
                        "target_intensities"
                    ),
                    target_intensities=run_config.match_points_kwargs.get(
                        "source_intensities"
                    ),
                    prior_transform=inv_prior_transform,
                )
                reverse_scores = reverse_scores.transpose()
            end = timer()
            scores_info["seconds"] = end - start
            write_scores(scores_file, scores)
            scores_info["scores_file"] = scores_file.name
            scores_info.update({f"algorithm_{k}": v for k, v in info.items()})
            if reverse_info is not None and reverse_scores is not None:
                scores_info.update(
                    {f"reverse_algorithm_{k}": v for k, v in reverse_info.items()}
                )
                if reverse_scores_file is not None:
                    write_scores(reverse_scores_file, reverse_scores)
                    scores_info["reverse_scores_file"] = reverse_scores_file.name
        except Exception as e:
            scores_info["error"] = str(e)
    return scores_info, scores, reverse_scores


def run_sequential(
    run_configs: Iterable[RunConfig],
    scores_dir: str | PathLike,
    offset: int = 0,
) -> Generator[
    tuple[dict[str, Any], xr.DataArray | None, xr.DataArray | None],
    None,
    pd.DataFrame,
]:
    scores_dir = Path(scores_dir)
    scores_dir.mkdir(exist_ok=True)
    scores_infos = []
    for i, run_config in enumerate(run_configs, start=offset):
        scores_info, scores, reverse_scores = run(
            run_config,
            scores_dir / f"scores{i:06d}.nc",
            reverse_scores_file=scores_dir / f"scores{i:06d}_reverse.nc",
        )
        scores_infos.append(scores_info)
        yield scores_info, scores, reverse_scores
    return pd.DataFrame(data=scores_infos)


def run_parallel(
    run_configs: Iterable[RunConfig],
    scores_dir: str | PathLike,
    offset: int = 0,
    n_processes: int | None = None,
    queue_size: int | None = None,
    worker_timeout: int = 1,
) -> Generator[RunConfig, None, pd.DataFrame]:
    """Run benchmark configurations in parallel worker processes.

    Note that the memory requirements of a single run can be substantial (e.g.
    several GB for Spellmatch with thousands of cells), so ``n_processes`` (default:
    number of CPUs) may have to be reduced to the available memory.

    ``worker_timeout`` is the interval (in seconds) for checking whether worker
    processes are still alive; if worker processes terminate unexpectedly (e.g. killed
    by the out-of-memory killer), a ``RuntimeError`` is raised instead of blocking.
    """
    scores_dir = Path(scores_dir)
    scores_dir.mkdir(exist_ok=True)
    if n_processes is None:
        n_processes = cpu_count()
    if queue_size is None:
        queue_size = n_processes
    run_config_queue = mp.Queue(maxsize=queue_size)
    scores_info_queue = mp.Queue()
    workers = [
        _WorkerProcess(
            run_config_queue,
            scores_info_queue,
            scores_dir,
            daemon=True,
            name=f"W{worker_number:03d}",
        )
        for worker_number in range(n_processes)
    ]
    for worker in workers:
        worker.start()
    n = 0
    for i, run_config in enumerate(run_configs, start=offset):
        _put(run_config_queue, (i, run_config), workers, worker_timeout)
        yield run_config
        n += 1
    for _ in workers:
        _put(run_config_queue, None, workers, worker_timeout)
    # results are collected before joining the workers, as a process that has put
    # items on a queue only terminates once they have been consumed
    scores_infos = [_get(scores_info_queue, workers, worker_timeout) for _ in range(n)]
    for worker in workers:
        worker.join()
    return pd.DataFrame(data=scores_infos)


def _put(q: mp.Queue, item: Any, workers: list[mp.Process], timeout: int) -> None:
    while True:
        try:
            q.put(item, timeout=timeout)
            return
        except queue.Full:
            if not any(worker.is_alive() for worker in workers):
                raise RuntimeError("Worker processes terminated unexpectedly") from None


def _get(q: mp.Queue, workers: list[mp.Process], timeout: int) -> Any:
    while True:
        # checked before waiting, as terminated workers may have sent results
        workers_alive = any(worker.is_alive() for worker in workers)
        try:
            return q.get(timeout=timeout)
        except queue.Empty:
            if not workers_alive:
                raise RuntimeError("Worker processes terminated unexpectedly") from None


# defined at module level to support the spawn/forkserver start methods
class _WorkerProcess(mp.Process):
    def __init__(
        self,
        run_config_queue: mp.Queue,
        scores_info_queue: mp.Queue,
        scores_dir: Path,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.run_config_queue = run_config_queue
        self.scores_info_queue = scores_info_queue
        self.scores_dir = scores_dir

    def run(self) -> None:
        # each worker receives one None (sentinel) after all run configurations
        for i, run_config in iter(self.run_config_queue.get, None):
            try:
                scores_info, _, _ = run(
                    run_config,
                    self.scores_dir / f"scores{i:06d}.nc",
                    reverse_scores_file=self.scores_dir / f"scores{i:06d}_reverse.nc",
                )
            except Exception as e:  # exactly one result per run configuration
                scores_info = {**run_config.info, "error": str(e)}
            self.scores_info_queue.put(scores_info)
