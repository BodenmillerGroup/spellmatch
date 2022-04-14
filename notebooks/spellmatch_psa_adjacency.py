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
# # Semi-synthetic Spellmatch adjacency parameter sensitivity analysis
#
# - Hand-picked images from Jackson & Fischer et al.
# - Fixed simutome parameters, 1 section per image
# - Spellmatch only
#     - Fixed similarity/prior weights
#     - Varying adjancy radii

# %%
import logging
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from spellmatch import logger
from spellmatch.assignment import assign
from spellmatch.benchmark.metrics import default_metrics
from spellmatch.benchmark.semisynthetic import (
    AlgorithmConfig,
    SemisyntheticBenchmark,
    SemisyntheticBenchmarkConfig,
)

try:
    from IPython.core.getipython import get_ipython
    in_ipython = get_ipython() is not None
except ImportError:
    get_ipython = None
    in_ipython = False

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
    simutome_kwargs={
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
    },
    simutome_param_grid={},
    n_simutome_sections=1,
    algorithm_configs={
        "spellmatch": AlgorithmConfig(
            algorithm_name="spellmatch",
            algorithm_kwargs={

                "filter_outliers": False,
                "intensity_transform": "numpy.arcsinh",
                "alpha": 0.7,  # TODO
                "spatial_cdist_prior_thres": 25.0,  # TODO
                "max_spatial_cdist": 50.0,
                "degree_weight": 1.0,  # TODO
                "degree_cdiff_thres": 3,  # TODO
                "intensity_weight": 1.0,  # TODO
                "intensity_interp_lmd": 1.0,  # TODO
                "intensity_shared_pca_n_components": 15,  # TODO
                "distance_weight": 1.0,  # TODO
                "distance_cdiff_thres": 5.0,  # TODO
                "scores_tol": 1e-6,
                "require_convergence": True,
                "require_opt_convergence": True,
            },
            algorithm_param_grid={
                "graph": [
                    {
                        "adj_radius": 12,
                    },
                    {
                        "adj_radius": 15,
                    },
                    {
                        "adj_radius": 18,
                    },
                ],
            },
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
benchmark = SemisyntheticBenchmark(benchmark_config, "spellmatch_psa_adjacency")
benchmark.save()

# %%
logger.setLevel(logging.INFO)
logger_file_handler = logging.FileHandler(
    benchmark.benchmark_dir / "spellmatch.log", mode="w"
)
logger_file_handler.setFormatter(
    logging.Formatter(fmt="%(asctime)s %(levelname)s %(name)s - %(message)s")
)
logger.addHandler(logger_file_handler)

# %%
n_batches = 1
batch_index = 0
if not in_ipython:
    parser = ArgumentParser()
    parser.add_argument("n_batches", type=int)
    parser.add_argument("batch_index", type=int)
    args = parser.parse_args()
    n_batches = args["n_batches"]
    batch_index = args["batch_index"]

for info, scores in tqdm(
    benchmark.run(
        source_points_dir,
        source_intensities_dir=source_intensities_dir,
        source_clusters_dir=source_clusters_dir,
        batch_index=batch_index,
        n_batches=n_batches,
    ),
    total=benchmark.get_run_length(n_batches),
):
    pass

# %%
for result in tqdm(
    benchmark.evaluate(assignment_functions, metric_functions),
    total=benchmark.get_evaluation_length(assignment_functions, metric_functions),
):
    pass
