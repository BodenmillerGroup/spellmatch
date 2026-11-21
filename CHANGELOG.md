# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Require Python 3.13 or newer
- Update dependencies to current versions (numpy 2, pandas 2.3+, pydantic 2, ...)
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

### Fixed

- Parallel benchmark runs with the spawn/forkserver multiprocessing start methods
  (default on Windows, macOS and, as of Python 3.14, Linux)
- Spellmatch: shared markers without variance no longer cause NaN errors when computing
  the intensity cross-distance (such markers are excluded with a warning)
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
  (e.g. `11.0`); the reported `lambda` is `None` if intensities are not used
  (previously, the raw parameter value was reported)
- Spellmatch: no crash with one shared marker, and no degenerate intensity
  cross-distances with two shared markers (the SVD is skipped for up to two shared
  markers)

## [0.1.0] - 2022-05-19

Initial release

[Unreleased]: https://github.com/BodenmillerGroup/spellmatch/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/BodenmillerGroup/spellmatch/releases/tag/v0.1.0
