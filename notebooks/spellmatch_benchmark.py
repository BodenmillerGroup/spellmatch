# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Semi-synthetic Spellmatch benchmark
#
# - Hand-picked images from Jackson & Fischer et al.
# - Varying simutome parameters, 1 section per image
# - Many algorithms, including Spellmatch
#     - Fixed similarity/prior weights
#     - Fixed adjacency radii

# %%
import logging
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from spellmatch.assignment import assign
from spellmatch.benchmark.metrics import default_metrics
from spellmatch.benchmark.semisynthetic import (
    SemisyntheticBenchmark,
    SemisyntheticBenchmarkConfig,
)

# %%
source_points_dir = "../data/jackson_fischer_2020/source_points"
source_intensities_dir = "../data/jackson_fischer_2020/source_intensities"
source_clusters_dir = "../data/jackson_fischer_2020/source_clusters"

benchmark_config = SemisyntheticBenchmarkConfig(
    source_points_file_names=[
        f.name for f in sorted(Path(source_points_dir).glob("*.csv"))
    ],
    source_intensities_file_names=[
        f.name for f in sorted(Path(source_intensities_dir).glob("*.csv"))
    ],
    source_clusters_file_names=[
        f.name for f in sorted(Path(source_clusters_dir).glob("*.csv"))
    ],
    simutome_kwargs={},  # TODO
    simutome_param_grid={},  # TODO
    n_simutome_sections=1,
    algorithm_configs={
        "spellmatch": SemisyntheticBenchmarkConfig.AlgorithmConfig(
            algorithm_name="spellmatch",
            algorithm_kwargs={},  # TODO
            algorithm_param_grid={},  # TODO
        ),
        "icp": SemisyntheticBenchmarkConfig.AlgorithmConfig(
            algorithm_name="icp",
            algorithm_kwargs={},  # TODO
            algorithm_param_grid={},  # TODO
        ),
        "rigid_cpd": SemisyntheticBenchmarkConfig.AlgorithmConfig(
            algorithm_name="rigid_cpd",
            algorithm_kwargs={},  # TODO
            algorithm_param_grid={},  # TODO
        ),
    },
    seed=123,
)

assignment_functions = {
    "min_score_q25_intersect": partial(
        assign, min_score_quantile=0.25, direction="intersect", as_matrix=True
    ),
    "min_score_q25_union": partial(
        assign, min_score_quantile=0.25, direction="union", as_matrix=True
    ),
    "max_only_intersect": partial(
        assign, max_only=True, direction="intersect", as_matrix=True
    ),
    "max_only_union": partial(
        assign, max_only=True, direction="union", as_matrix=True
    ),
    "linear_sum_forward": partial(
        assign, linear_sum=True, direction="forward", as_matrix=True
    ),
}
metric_functions = default_metrics

# %%
parser = ArgumentParser()
parser.add_argument("--path", type=str, default="spellmatch_benchmark")
parser.add_argument("--batch", type=int, default=0)
parser.add_argument("--nbatch", type=int, default=1)
parser.add_argument("--nproc", type=int, default=None)
benchmark_args, _ = parser.parse_known_args()

# %%
benchmark_dir = Path(benchmark_args.path)
benchmark_dir.mkdir(exist_ok=True)
logging.basicConfig(
    filename=benchmark_dir / "benchmark.log",
    filemode="w",
    format="[%(processName)-4s] %(asctime)s %(levelname)s %(name)s - %(message)s",
    level=logging.INFO,
    force=True,
)

# %%
benchmark = SemisyntheticBenchmark(benchmark_dir, benchmark_config)
benchmark.save()

# %%
for run_config in tqdm(
    benchmark.run_parallel(
        source_points_dir,
        source_intensities_dir=source_intensities_dir,
        source_clusters_dir=source_clusters_dir,
        batch_index=benchmark_args.batch,
        n_batches=benchmark_args.nbatch,
        n_processes=benchmark_args.nproc,
    ),
    total=benchmark.get_run_length(benchmark_args.nbatch),
):
    pass

# %%
for result in tqdm(
    benchmark.evaluate(assignment_functions, metric_functions),
    total=benchmark.get_evaluation_length(assignment_functions, metric_functions),
):
    pass