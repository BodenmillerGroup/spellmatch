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
# # Semi-synthetic Spellmatch weighting parameter sensitivity analysis
#
# - Hand-picked images from Jackson & Fischer et al.
# - Fixed simutome parameters, 1 section per image
# - Spellmatch only
#     - Fixed adjancy radius of $15 \mu m$
#     - Varying similarity/prior weights

# %%
import logging
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from tqdm.auto import tqdm

from spellmatch import logger
from spellmatch.assignment import assign
from spellmatch.benchmark import AlgorithmConfig, Benchmark
from spellmatch.benchmark.metrics import default_metrics
from spellmatch.io import write_scores
from spellmatch.matching.algorithms.spellmatch import Spellmatch

# %%
source_points_dir = "../data/jackson_fischer_2020/source_points"
source_intensities_dir = "../data/jackson_fischer_2020/source_intensities"
source_clusters_dir = "../data/jackson_fischer_2020/source_clusters"

simutome_kwargs = {
    # assume minor mis-alignment
    "image_rotation": 2.0 * np.pi / 180,
    "image_translation": (1.0, 3.0),
    # see simutome_parameters.ipynb
    "exclude_cells": True,
    "section_thickness": 2.0,
    "cell_diameter_mean": 7.931,
    "cell_diameter_std": 1.768,
    # see simutome_parameters.ipynb
    "displace_cells": True,
    "cell_displacement_mean": 0.067,
    "cell_displacement_var": 1.010,
}
simutome_param_grid = ParameterGrid({})
num_sections = 1

algorithm_configs = {
    "spellmatch": AlgorithmConfig(
        Spellmatch,
        algorithm_kwargs={
            "adj_radius": 15,
            "filter_outliers": False,
            "intensity_transform": "numpy.arcsinh",
            "max_spatial_cdist": 50.0,
            "scores_tol": 1e-6,
            "require_convergence": True,
            "require_opt_convergence": True,
        },
        algorithm_param_grid=ParameterGrid(
            {
                "prior": [
                    {
                        "alpha": 0.8,
                        "spatial_cdist_prior_thres": 25.0,
                    },
                    {
                        "alpha": 0.7,
                        "spatial_cdist_prior_thres": 25.0,
                    },
                    {
                        "alpha": 0.9,
                        "spatial_cdist_prior_thres": 25.0,
                    },
                    {
                        "alpha": 1.0,
                        "spatial_cdist_prior_thres": 25.0,
                    },
                ],
                "degrees": [
                    {
                        "degree_weight": 1.0,
                        "degree_cdiff_thres": 3,
                    },
                    {
                        "degree_weight": 0.1,
                        "degree_cdiff_thres": 3,
                    },
                    {
                        "degree_weight": 10.0,
                        "degree_cdiff_thres": 3,
                    },
                    {
                        "degree_weight": 0.0,
                    },
                ],
                "intensity": [
                    {
                        "intensity_weight": 1.0,
                        "intensity_interp_lmd": 1.0,
                        "intensity_shared_pca_n_components": 15,
                    },
                    {
                        "intensity_weight": 0.1,
                        "intensity_interp_lmd": 1.0,
                        "intensity_shared_pca_n_components": 15,
                    },
                    {
                        "intensity_weight": 10.0,
                        "intensity_interp_lmd": 1.0,
                        "intensity_shared_pca_n_components": 15,
                    },
                    {
                        "intensity_weight": 0.0,
                    },
                ],
                "distances": [
                    {
                        "distance_weight": 1.0,
                        "distance_cdiff_thres": 5.0,
                    },
                    {
                        "distance_weight": 0.1,
                        "distance_cdiff_thres": 5.0,
                    },
                    {
                        "distance_weight": 10.0,
                        "distance_cdiff_thres": 5.0,
                    },
                    {
                        "distance_weight": 0.0,
                    },
                ],
            }
        ),
        assignment_functions={
            "min_score_q25_union": partial(assign, min_score_quantile=0.25, direction="union", as_matrix=True),
            "max_only_intersect": partial(assign, max_only=True, direction="intersect", as_matrix=True),
            "linear_sum_forward": partial(assign, linear_sum=True, direction="forward", as_matrix=True),
        }
    ),
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

benchmark_dir = Path("spellmatch_psa_weighting")
scores_dir = benchmark_dir / "scores"
scores_dir.mkdir(exist_ok=True, parents=True)

logger.setLevel(logging.INFO)
logger_handler = logging.FileHandler(benchmark_dir / "spellmatch.log", mode="w")
logger_handler.setFormatter(logging.Formatter(fmt="%(asctime)s %(levelname)s %(name)s - %(message)s"))
logger.addHandler(logger_handler)

infos = []
all_results = []
save_interval = 10
for i, (info, scores, results) in enumerate(tqdm(benchmark)):
    infos.append(info)
    if scores is not None:
        info["scores_file"] = f"scores{i:06d}.nc"
        write_scores(scores_dir / info["scores_file"], scores)
    if results is not None:
        for j, result in enumerate(results):
            results[j] = {**info, **result}
        all_results += results
    if (i + 1) % save_interval == 0 or (i + 1) >= len(benchmark):
        pd.DataFrame(data=infos).to_csv(benchmark_dir / "infos.csv", index=False)
        pd.DataFrame(data=all_results).to_csv(benchmark_dir / "results.csv", index=False)
