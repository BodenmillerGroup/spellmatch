# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Require Python 3.13 or newer
- Update dependencies to current versions (numpy 2, pandas 2.3+, pydantic 2, ...)
- Benchmark configurations (`SemisyntheticBenchmarkConfig`) are validated using
  pydantic 2, which is stricter in type coercion (e.g. numbers are no longer converted
  to strings); hand-written configurations may need to be adapted
- Make napari/Qt (`gui`) and probreg (`probreg`) optional dependencies; install
  `spellmatch[all]` for the previous set of features
- Migrate packaging to `pyproject.toml` (hatchling, hatch-vcs) and development tooling
  to uv and ruff
- Publish to PyPI using trusted publishing
- Replace the conda `environment.yml` with the `analysis` uv dependency group
- Replace the unmaintained click-log dependency (no change in CLI behavior)
- Spellmatch: `max_spatial_cdist` (d_max) is now a hard constraint; cell pairs farther
  apart are excluded from the optimization and receive a score of zero (previously,
  it only affected the initialization and had no effect on converged scores)
- Spellmatch: new parameters `intensity_shared_metric` (`reconstructed_correlation`,
  `cosine`, `correlation`) and `intensity_all_metric` (`cosine`, `correlation`) to
  select the intensity cross-distance metrics. **The defaults changed**: shared-marker
  intensities are now compared using the Pearson correlation across markers between
  rank-k SVD reconstructions (previously: across SVD components), and all-marker
  intensities using the cosine distance between CCA embeddings (previously: Pearson
  correlation across CCA components). Set both parameters to `correlation` to restore
  the previous behavior.
- Spellmatch: without a prior transform, the spatial prior (`spatial_cdist_prior_thres`),
  the maximum spatial cross-distance (`max_spatial_cdist`) and the intensity
  cross-distance from all markers (CCA fitted on spatially close cells) are used from
  the first estimated transform on; previously, the untransformed cell coordinates were
  used, i.e. the images were implicitly assumed to be aligned. Pass an identity
  transform as prior transform to restore the previous behavior
- Spellmatch: `alpha` must be smaller than 1; without a spatial prior (`alpha=1`), the
  fixed point iteration is ill-defined (documented, no change in behavior)
- Spellmatch: less verbose log output; per-step and per-λ messages are logged at the
  `DEBUG` level (`--verbosity DEBUG`), and repeated warnings once per matching run
- Iterative algorithms: `scores_tol` is now a tolerance for the *relative* change in
  scores between iterations (previously absolute, which depended on the number of
  cells); Spellmatch now defaults to `scores_tol=1e-3` (previously `None`, i.e., always
  `max_iter` iterations)
- Require scikit-image 0.26 or newer (`Transform.from_estimate`), numpy 2.4.2 or newer
  except on macOS (the wheels of numpy 2.3.2–2.4.1 bundle OpenBLAS 0.3.30, which
  deadlocks in linear algebra routines after a fork, e.g. `run_parallel` with the fork
  start method), and versions with wheels for Python 3.14 (numpy 2.3.2, pandas 2.3.3,
  pydantic 2.12, PyYAML 6.0.3, scikit-learn 1.7.2, scipy 1.16.1, shapely 2.1.2); the
  lower bounds are now tested in CI

### Fixed

- Parallel benchmark runs with the spawn/forkserver multiprocessing start methods
  (default on Windows, macOS and, as of Python 3.14, Linux)
- Spellmatch: shared markers without variance no longer cause NaN errors when computing
  the intensity cross-distance (such markers are excluded with a warning)
- Spellmatch: informative errors instead of NaN scores if no cell pairs lie within
  `spatial_cdist_prior_thres` or `max_spatial_cdist`
- Spellmatch: the degrees of the product graph (used for normalization) no longer
  overflow for cells with many neighbors (8-bit integer products of cell degrees)
- Spellmatch: no crash when the number of SVD components for the shared-marker
  intensity cross-distance equals the number of shared markers
- Spellmatch: the degree cross-distance no longer wraps around when the source cell has
  fewer neighbors than the target cell, or more than 255 neighbors (8-bit integer cell
  degrees). **This changes results** for all runs with `degree_weight > 0`
- Spellmatch: without outlier filtering, the all-marker intensity cross-distance is now
  updated in every iteration, and graphs and graph cross-distances are updated for
  similarity/affine transforms. **This changes results** for such runs with
  `intensity_interp_lmd` < 1 or non-rigid transform models
- Spellmatch: no crash with fewer cells than `intensity_all_cca_fit_k_closest` or
  `intensity_all_cca_n_components`, or with `intensity_all_cca_fit_k_most_certain` >
  `intensity_all_cca_fit_k_closest`
- Spellmatch: no crash for floating point values of `intensity_interp_lmd` larger than 1
  (e.g. `11.0`); the reported `lambda` is the effective λ (`None` if intensities are not
  used; previously, the raw parameter value was reported)
- Spellmatch: no crash with one shared marker, and no degenerate intensity
  cross-distances with two shared markers (the SVD is skipped for up to two shared
  markers)
- Spellmatch: no NaN or degenerate intensity cross-distances with only one or two SVD/CCA
  components; differences thresholded at `intensity_1d_cdiff_thres_sd` (new
  parameter) standard deviations, or cosine distances, are used instead. CCA
  convergence warnings are logged with a hint to increase `cca_max_iter`
- Iterative algorithms: no crash with outlier filtering (default) and no prior transform
- Iterative algorithms: no crash with fewer cells than `transform_estim_k_best`, or with
  `max_margin` transform estimation and a single target cell
- Matching: no crash for source and target masks/point sets with identical names (e.g.
  identically named mask files in different directories); the score dimensions are now
  suffixed with ` (source)`/` (target)` in this case
- Matching: intensities are now aligned to points by cell label instead of row position
  without outlier filtering (Python API)
- probreg algorithms: no crash when cupy is installed without a working CUDA runtime
- `assign`: no crash for maximum assignment in reverse direction (`--max
  --direction=reverse`)
- `assign`: no crash with margin thresholds and a single source or target cell
- `assign`: no crash in directory mode without `--validate`
- `assign --show`: no crash with pandas copy-on-write (read-only arrays); target labels
  that do not fit the data type of the source mask are no longer displayed in the wrong
  color
- `assign --validate`: validation assignments are now matched by column position
  (source, target); previously, they were merged on shared column names, which could
  over-count recovered cell pairs
- CLI: keyword arguments (e.g. `--algorithm-args`) are parsed using the safe YAML loader
- `match --denoise-*/--clip-*/--blur-*`: image filters were also applied across channels
  (signal bled into neighboring channels), and the clipping quantile was pooled across
  channels; filters and clipping are now applied to each channel separately
- Anisotropic pixel sizes were applied in the wrong axis order to cell coordinates (3D:
  x and z swapped) and bounding boxes
- `register features`: keypoints of non-square images were centered using swapped axes
  (spurious translation), `keep_matches_frac` kept arbitrary instead of the best matches,
  the pixel size (`--source-scale`/`--target-scale`) was ignored (transform in pixel
  instead of physical units), failed RANSAC fits silently resulted in an identity
  transform, and the default `ransac_kwargs=None` crashed (Python API)
- `register interactive`: single-channel images were displayed with an offset relative
  to the mask; layers are now displayed in the same (physical) coordinates as cell
  centroids; Save is only enabled with enough cell pairs for the transform type
- `register intensities`: metric sampling (`RANDOM`/`REGULAR` sampling strategies) is now
  reproducible by default (`sampling_seed=42`; set to `0` for wall-clock seeding)
- Semi-synthetic benchmark: `algorithm_is_directed` was ignored when generating run
  configurations, so reverse matching was never performed for directed algorithms
  (e.g. ICP, CPD)
- Semi-synthetic benchmark: `SemisyntheticBenchmark.load` failed (YAML loader, swapped
  arguments, and `config.yml` instead of `config.yaml` written by `save`)
- Semi-synthetic benchmark: missing reverse scores files are detected robustly
- Benchmark metrics: the entropy metric could return NaN, normalizing metrics modified
  the scores in-place (shared with subsequent assignments/metrics), and the margin
  metric failed for a single source or target cell
- `run_parallel`: deadlock when many results were queued, and indefinite blocking when
  worker processes terminated unexpectedly (now raises a `RuntimeError`)
- GUI features (`register interactive`, `--show`) now report that the optional `gui`
  dependencies are missing, instead of failing with a `ModuleNotFoundError`
- A broken probreg installation (e.g. missing system libraries required by open3d) no
  longer breaks the entire CLI; the probreg algorithms are skipped with a warning

## [0.1.0] - 2022-05-19

Initial release

[Unreleased]: https://github.com/BodenmillerGroup/spellmatch/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/BodenmillerGroup/spellmatch/releases/tag/v0.1.0
