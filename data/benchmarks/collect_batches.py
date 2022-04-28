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

# %%
import shutil
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

# %%
parser = ArgumentParser()
parser.add_argument("--path", type=str, default="")
args, _ = parser.parse_known_args()
assert args.path, "path argument"

# %%
results_dir = Path(args.path)
batch_dirs = sorted(results_dir.glob("batch*"))

scores_dir = results_dir / "scores"
scores_dir.mkdir()
log_dir = results_dir / "logs"
log_dir.mkdir()

scores_infos = []
results_infos = []
for batch_dir in tqdm(batch_dirs):
    config_file = batch_dir / "config.yaml"
    shutil.copy2(config_file, results_dir)
    for scores_file in sorted((batch_dir / "scores").glob("*.nc")):
        assert not (scores_dir / scores_file.name).exists()
        shutil.copy2(scores_file, scores_dir / scores_file.name)
    log_file = batch_dir / "benchmark.log"
    shutil.copy2(log_file, log_dir / f"{batch_dir.name}.log")
    scores_info = pd.read_csv(batch_dir / "scores.csv")
    scores_infos.append(scores_info)
    results_info = pd.read_csv(batch_dir / "results.csv")
    results_infos.append(results_info)
pd.concat(scores_infos).to_csv(results_dir / "scores.csv", index=False)
pd.concat(results_infos).to_csv(results_dir / "results.csv", index=False)
