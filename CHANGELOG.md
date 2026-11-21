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
- Spellmatch: new parameters `intensity_shared_metric` (`correlation`, `cosine`,
  `reconstructed_correlation`) and `intensity_all_metric` (`correlation`, `cosine`) to
  select the intensity cross-distance metrics (defaults unchanged)

### Fixed

- Parallel benchmark runs with the spawn/forkserver multiprocessing start methods
  (default on Windows, macOS and, as of Python 3.14, Linux)
- Spellmatch: shared markers without variance no longer cause NaN errors when computing
  the intensity cross-distance (such markers are excluded with a warning)
- Spellmatch: the degrees of the product graph (used for normalization) no longer
  overflow for cells with many neighbors (8-bit integer products of cell degrees)
- Spellmatch: no crash when the number of SVD components for the shared-marker
  intensity cross-distance equals the number of shared markers

## [0.1.0] - 2022-05-19

Initial release

[Unreleased]: https://github.com/BodenmillerGroup/spellmatch/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/BodenmillerGroup/spellmatch/releases/tag/v0.1.0
