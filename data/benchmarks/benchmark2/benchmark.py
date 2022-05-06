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
# # Semi-synthetic algorithm benchmark
#
# - 32 randomly selected images from Jackson & Fischer et al.
# - Varying simutome mis-alignment parameters, 1 section per image
# - All algorithms, including Spellmatch:
#     - Cell exclusion and cell swapping
#     - Fixed adjancy radius of $15 \mu m$
#     - Fixed similarity weights
#         - $w_\text{degree} = 1$
#         - $w_\text{intensity} = 1$
#         - $w_\text{distance} = 10$
#     - Fixed prior weight of $\alpha = 0.8$

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
points_dir = "../../datasets/jackson_fischer_2020/points32"
intensities_dir = "../../datasets/jackson_fischer_2020/intensities32"
clusters_dir = "../../datasets/jackson_fischer_2020/clusters32"

benchmark_config = SemisyntheticBenchmarkConfig(
    points_file_names=[f.name for f in sorted(Path(points_dir).glob("*.csv"))],
    intensities_file_names=[
        f.name for f in sorted(Path(intensities_dir).glob("*.csv"))
    ],
    clusters_file_names=[f.name for f in sorted(Path(clusters_dir).glob("*.csv"))],
    simutome_kwargs={
        # do not occlude images (assume images to be co-registered & cropped)
        "image_occlusion": 0.0,
        # exclude cells according to parameter estimates from Kuett et al.
        "exclude_cells": True,
        "exclude_cells_swap": 0.5,
        "section_thickness": 2.0,
        "cell_diameter_mean": 7.931,
        "cell_diameter_std": 1.768,
        # displace cells according to parameter estimates from Kuett et al.
        "displace_cells": True,
        "cell_displacement_mean": 0.067,
        "cell_displacement_var": 1.010,
        # misalignment: do not scale or shear the image (only rotate/translate)
        "image_scale": (1.0, 1.0),
        "image_shear": 0.0,
        # do not split cells (assume perfect segmentation)
        "cell_division_probab": 0.0,
        "cell_division_dist_mean": None,
        "cell_division_dist_std": None,
    },
    simutome_param_grid={
        "misalignment_rotation": [
            {"image_rotation": 0.0},
            {"image_rotation": 2.0 * np.pi / 180.0},
            {"image_rotation": 4.0 * np.pi / 180.0},
        ],
        "misalignment_translation": [
            {"image_translation": (0.0, 0.0)},
            {"image_translation": (10.0, 0.0)},
            {"image_translation": (20.0, 0.0)},
        ],
    },
    n_simutome_sections=1,
    algorithm_configs={
        "icp": SemisyntheticBenchmarkConfig.AlgorithmConfig(
            algorithm_name="icp",
            algorithm_kwargs={
                "scores_tol": 1e-6,
                "filter_outliers": False,
                "max_dist": 25.0,
                "min_change": 1e-9,
            },
            algorithm_param_grid={},
            algorithm_is_directed=True,
        ),
        "rigid_cpd": SemisyntheticBenchmarkConfig.AlgorithmConfig(
            algorithm_name="rigid_cpd",
            algorithm_kwargs={
                "max_dist": 25.0,
                "w": 0.25,
                "maxiter": 500,
                "tol": 1e-6,
                "update_scale": False,
            },
            algorithm_param_grid={},
            algorithm_is_directed=True,
        ),
        "spellmatch": SemisyntheticBenchmarkConfig.AlgorithmConfig(
            algorithm_name="spellmatch",
            algorithm_kwargs={
                "adj_radius": 15,
                "filter_outliers": False,
                "intensity_transform": "numpy.arcsinh",
                "max_spatial_cdist": 25.0,
                "scores_tol": 1e-6,
                "alpha": 0.8,
                "spatial_cdist_prior_thres": 25.0,
                "degree_weight": 1.0,
                "degree_cdiff_thres": 3,
                "intensity_weight": 1.0,
                "intensity_interp_lmd": 1.0,
                "intensity_shared_pca_n_components": 15,
                "distance_weight": 10.0,
                "distance_cdiff_thres": 5.0,
            },
            algorithm_param_grid={},
        ),
    },
    seed=123,
)

assignment_functions = {
    "linear_sum": partial(
        assign,
        linear_sum_assignment=True,
        as_matrix=True,
    ),
    "max_intersect": partial(
        assign,
        max_assignment=True,
        assignment_direction="intersect",
        as_matrix=True,
    ),
    "max_union": partial(
        assign,
        max_assignment=True,
        assignment_direction="union",
        as_matrix=True,
    ),
    "max_union_thresQ05": partial(
        assign,
        max_assignment=True,
        assignment_direction="union",
        min_post_assignment_score_quantile=0.05,
        as_matrix=True,
    ),
    "max_union_thresQ15": partial(
        assign,
        max_assignment=True,
        assignment_direction="union",
        min_post_assignment_score_quantile=0.15,
        as_matrix=True,
    ),
    "max_union_thresQ25": partial(
        assign,
        max_assignment=True,
        assignment_direction="union",
        min_post_assignment_score_quantile=0.25,
        as_matrix=True,
    ),
}
metric_functions = default_metrics

# %%
parser = ArgumentParser()
parser.add_argument("--batch", type=int, default=0)
parser.add_argument("--nbatch", type=int, default=1)
args, _ = parser.parse_known_args()

# %%
results_dir = Path("results")
if args.nbatch > 1:
    results_dir /= f"batch{args.batch:03d}"
results_dir.mkdir(parents=True, exist_ok=True)
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
# RigidCPD does not seem to support multiprocessing
# see https://github.com/neka-nat/probreg/issues/90
for run_config in tqdm(
    benchmark.run_sequential(
        points_dir,
        intensities_dir=intensities_dir,
        clusters_dir=clusters_dir,
        batch_index=args.batch,
        n_batches=args.nbatch,
    ),
    total=benchmark.get_run_length(args.nbatch),
):
    pass

# %%
# import numpy as np
# import pandas as pd

# scores_info = pd.read_csv(results_dir / "scores.csv")
# scores_info["error"] = np.nan
# for i, scores_file_name in enumerate(scores_info["scores_file"].tolist()):
#     if not (results_dir / "scores" / scores_file_name).exists():
#         scores_info.loc[i, "seconds"] = np.nan
#         scores_info.loc[i, "scores_file"] = np.nan
#         scores_info.loc[i, "error"] = "unknown"
# scores_info.to_csv(results_dir / "scores.csv")

# benchmark.scores_info = scores_info

# %%
for result in tqdm(
    benchmark.evaluate(assignment_functions, metric_functions),
    total=benchmark.get_evaluation_length(assignment_functions, metric_functions),
):
    pass