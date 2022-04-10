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
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm

from spellmatch.assignment import assign
from spellmatch.benchmark import AlgorithmConfig, Benchmark
from spellmatch.benchmark.metrics import default_metrics
from spellmatch.io import write_scores
from spellmatch.matching.algorithms.spellmatch import Spellmatch

# %%
source_points_dir = "../data/jackson_fischer_2020/source_points"
source_intensities_dir = "../data/jackson_fischer_2020/source_intensities"
source_clusters_dir = "../data/jackson_fischer_2020/source_clusters"

simutome_kwargs = {}  # TODO
simutome_param_grid = ParameterGrid({})  # TODO
num_sections = 1

algorithm_configs = {
    "spellmatch": AlgorithmConfig(
        Spellmatch,
        algorithm_kwargs={},  # TODO
        algorithm_param_grid=ParameterGrid({}), # TODO
        assignment_functions={
            "min_score_q25_union": partial(assign, min_score_quantile=0.25, direction="union", as_matrix=True),
            "max_only_intersect": partial(assign, max_only=True, direction="intersect", as_matrix=True),
            "linear_sum_forward": partial(assign, linear_sum=True, direction="forward", as_matrix=True),
        }
    )
}

metric_functions = default_metrics

# %%
benchmark = Benchmark(
    source_points_files=sorted(Path(source_points_dir).glob("*.csv")),
    source_intensities_files=sorted(Path(source_intensities_dir).glob("*.csv")),
    source_clusters_files=sorted(Path(source_clusters_dir).glob("*.csv")),
    simutome_kwargs=simutome_kwargs,
    simutome_param_grid=simutome_param_grid,
    num_sections=num_sections,
    algorithm_configs=algorithm_configs,
    metric_functions=metric_functions,
    seed=123,
)

benchmark_dir = Path("spellmatch_benchmark")
scores_dir = benchmark_dir / "scores"
scores_dir.mkdir(exist_ok=True, parents=True)

infos = []
all_results = []
for i, (info, scores, results) in enumerate(tqdm(benchmark)):
    infos.append(info)
    if scores is not None:
        scores_file_name = f"scores{i:06d}.nc"
        write_scores(scores_dir / scores_file_name, scores)
    if results is not None:
        for i, result in enumerate(results):
            results[i] = {**info, **results}
        all_results += results
infos = pd.DataFrame(data=infos)
infos.to_csv(benchmark_dir / "infos.csv", index=False)
all_results = pd.DataFrame(data=all_results)
all_results.to_csv(benchmark_dir / "results.csv", index=False)
