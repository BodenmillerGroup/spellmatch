import queue
from collections.abc import Iterable
from multiprocessing import Event, JoinableQueue, Process, Queue
from os import PathLike, cpu_count
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Generator, Optional, Union

import pandas as pd
import xarray as xr
from pydantic import BaseModel

from ..io import write_scores
from ..matching.algorithms import PointsMatchingAlgorithm
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
    scores_info = run_config.info.copy()
    algorithm = None
    try:
        algorithm_types: list[
            type[PointsMatchingAlgorithm]
        ] = plugin_manager.hook.spellmatch_get_mask_matching_algorithm(
            name=run_config.algorithm_name
        )
        assert len(algorithm_types) == 1
        algorithm_type = algorithm_types[0]
        algorithm = algorithm_type(**run_config.algorithm_kwargs)
    except Exception as e:
        scores_info["error"] = str(e)
    scores = None
    if algorithm is not None:
        try:
            start = timer()
            scores = algorithm.match_points(**run_config.match_points_kwargs)
            end = timer()
            scores_info["seconds"] = end - start
            scores_info["scores_file"] = scores_file.name
            write_scores(scores_file, scores)
        except Exception as e:
            scores_info["error"] = str(e)
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
    worker_timeout: int = 1,
) -> Generator[RunConfig, None, pd.DataFrame]:
    class WorkerProcess(Process):
        def __init__(
            self,
            run_config_queue: JoinableQueue,
            scores_info_queue: Queue,
            scores_dir: Path,
            timeout: int,
            **kwargs: Any,
        ) -> None:
            super(WorkerProcess, self).__init__(**kwargs)
            self.run_config_queue = run_config_queue
            self.scores_info_queue = scores_info_queue
            self.scores_dir = scores_dir
            self.timeout = timeout
            self.stop_event = Event()

        def run(self) -> None:
            while not self.stop_event.is_set():
                try:
                    i: int
                    run_config: RunConfig
                    i, run_config = self.run_config_queue.get(timeout=self.timeout)
                    scores_file = self.scores_dir / f"scores{i:06d}.nc"
                    scores_info, scores = run(run_config, scores_file)
                    self.scores_info_queue.put(scores_info)
                    self.run_config_queue.task_done()
                except queue.Empty:
                    pass

    scores_dir = Path(scores_dir)
    scores_dir.mkdir(exist_ok=True)
    if n_processes is None:
        n_processes = cpu_count()
    if queue_size is None:
        queue_size = n_processes
    run_config_queue = JoinableQueue(maxsize=queue_size)
    scores_info_queue = Queue()
    workers = [
        WorkerProcess(
            run_config_queue,
            scores_info_queue,
            scores_dir,
            worker_timeout,
            daemon=True,
            name=f"W{worker_number:03d}",
        )
        for worker_number in range(n_processes)
    ]
    for worker in workers:
        worker.start()
    n = 0
    for i, run_config in enumerate(run_configs):
        run_config_queue.put((i, run_config))
        yield run_config
        n += 1
    run_config_queue.join()
    assert scores_info_queue.qsize() == n
    for worker in workers:
        worker.stop_event.set()
    return pd.DataFrame(data=[scores_info_queue.get() for _ in range(n)])
