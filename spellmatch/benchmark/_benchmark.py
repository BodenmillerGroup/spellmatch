from collections.abc import Iterable
from multiprocessing import Process, Queue
from os import PathLike, cpu_count
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Generator, Optional, Union

import pandas as pd
import xarray as xr
from pydantic import BaseModel

from ..io import write_scores
from ..matching.algorithms import (
    PointsMatchingAlgorithm,
    SpellmatchMatchingAlgorithmException,
)
from ..plugins import plugin_manager


class RunConfig(BaseModel):
    info: dict[str, Any]
    algorithm_name: str
    algorithm_kwargs: dict[str, Any]
    match_points_kwargs: dict[str, Any]


def run(
    run_config: RunConfig, scores_file: Union[str, PathLike]
) -> tuple[dict[str, Any], xr.DataArray]:
    scores_file = Path(scores_file)
    algorithm_types: list[
        type[PointsMatchingAlgorithm]
    ] = plugin_manager.hook.spellmatch_get_mask_matching_algorithm(
        name=run_config.algorithm_name
    )
    assert len(algorithm_types) == 1
    algorithm_type = algorithm_types[0]
    algorithm = algorithm_type(**run_config.algorithm_kwargs)
    start = timer()
    scores = None
    error = None
    try:
        scores = algorithm.match_points(**run_config.match_points_kwargs)
    except SpellmatchMatchingAlgorithmException as e:
        error = str(e)
    end = timer()
    scores_info = run_config.info.copy()
    if scores is not None:
        scores_info["scores_file"] = scores_file.name
        write_scores(scores_file, scores)
    scores_info["seconds"] = end - start
    scores_info["error"] = error
    return scores_info, scores


def run_sequential(
    run_configs: Iterable[RunConfig], scores_dir: Union[str, PathLike]
) -> Generator[tuple[dict[str, Any], xr.DataArray], None, pd.DataFrame]:
    scores_dir = Path(scores_dir)
    scores_dir.mkdir(exist_ok=True)
    scores_infos = []
    for i, run_config in enumerate(run_configs):
        scores_info, scores = run(run_config, scores_dir / f"scores{i:06d}.nc")
        scores_infos.append(scores_info)
        yield scores_info, scores
    return pd.DataFrame(data=scores_infos)


def run_parallel(
    run_configs: Iterable[RunConfig],
    scores_dir: Union[str, PathLike],
    n_processes: Optional[int] = None,
    queue_size: int = None,
) -> Generator[RunConfig, None, pd.DataFrame]:
    class AlgorithmProcess(Process):
        def __init__(
            self,
            run_config_queue: Queue,
            scores_info_queue: Queue,
            scores_dir: Path,
            timeout: int = 5,
        ) -> None:
            super(AlgorithmProcess, self).__init__(daemon=True)
            self.run_config_queue = run_config_queue
            self.scores_info_queue = scores_info_queue
            self.scores_dir = scores_dir
            self.timeout = timeout
            self.exit = False

        def run(self) -> None:
            while not self.exit or self.run_config_queue.qsize() > 0:
                try:
                    i: int
                    run_config: RunConfig
                    i, run_config = self.run_config_queue.get(timeout=self.timeout)
                    scores_file = self.scores_dir / f"scores{i:06d}.nc"
                    scores_info = run(run_config, scores_file)
                    self.scores_info_queue.put(scores_info)
                except self.run_config_queue.Empty:
                    pass

    scores_dir = Path(scores_dir)
    scores_dir.mkdir(exist_ok=True)
    if n_processes is None:
        n_processes = cpu_count()
    if queue_size is None:
        queue_size = n_processes
    run_config_queue = Queue(maxsize=queue_size)
    scores_info_queue = Queue()
    workers = [
        AlgorithmProcess(run_config_queue, scores_info_queue, scores_dir)
        for _ in range(n_processes)
    ]
    for worker in workers:
        worker.start()
    n = 0
    for i, run_config in enumerate(run_configs):
        run_config_queue.put((i, run_config))
        yield run_config
        n += 1
    for worker in workers:
        worker.exit = True
    for worker in workers:
        worker.join()
    assert scores_info_queue.qsize() == n
    return pd.DataFrame(data=[scores_info_queue.get() for _ in range(n)])
