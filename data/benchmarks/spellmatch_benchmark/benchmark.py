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
points_dir = "../datasets/jackson_fischer_2020/points"
intensities_dir = "../datasets/jackson_fischer_2020/intensities"
clusters_dir = "../datasets/jackson_fischer_2020/clusters"

benchmark_config = SemisyntheticBenchmarkConfig(
    points_file_names=[
        f.name for f in sorted(Path(points_dir).glob("*.csv"))
    ],
    intensities_file_names=[
        f.name for f in sorted(Path(intensities_dir).glob("*.csv"))
    ],
    clusters_file_names=[
        f.name for f in sorted(Path(clusters_dir).glob("*.csv"))
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
    "linear_sum": partial(
        assign, linear_sum=True, as_matrix=True
    ),
    "max_intersect": partial(
        assign, max=True, direction="intersect", as_matrix=True
    ),
    "max_union": partial(
        assign, max=True, direction="union", as_matrix=True
    ),
    "thresQ1_intersect": partial(
        assign, min_score_quantile=0.25, direction="intersect", as_matrix=True
    ),
    "thresQ1_union": partial(
        assign, min_score_quantile=0.25, direction="union", as_matrix=True
    ),
}
metric_functions = default_metrics

# %%
parser = ArgumentParser()
parser.add_argument("--batch", type=int, default=0)
parser.add_argument("--nbatch", type=int, default=1)
parser.add_argument("--nproc", type=int, default=None)
args, _ = parser.parse_known_args()

# %%
results_dir = Path("results")
results_dir.mkdir()
if args.nbatch > 1:
    results_dir /= f"batch{args.batch:03d}"
    results_dir.mkdir()
logging.basicConfig(
    filename=results_dir / "benchmark.log",
    filemode="w",
    format="[%(processName)-4s] %(asctime)s %(levelname)s %(name)s - %(message)s",
    level=logging.INFO,
    force=True,
)

# %%
benchmark = SemisyntheticBenchmark(results_dir, benchmark_config)
benchmark.save()

# %%
for run_config in tqdm(
    benchmark.run_parallel(
        points_dir,
        intensities_dir=intensities_dir,
        clusters_dir=clusters_dir,
        batch_index=args.batch,
        n_batches=args.nbatch,
        n_processes=args.nproc,
    ),
    total=benchmark.get_run_length(args.nbatch),
):
    pass

# %%
for result in tqdm(
    benchmark.evaluate(assignment_functions, metric_functions),
    total=benchmark.get_evaluation_length(assignment_functions, metric_functions),
):
    pass
