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

### Fixed

- Parallel benchmark runs with the spawn/forkserver multiprocessing start methods
  (default on Windows, macOS and, as of Python 3.14, Linux)

## [0.1.0] - 2022-05-19

Initial release

[Unreleased]: https://github.com/BodenmillerGroup/spellmatch/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/BodenmillerGroup/spellmatch/releases/tag/v0.1.0
