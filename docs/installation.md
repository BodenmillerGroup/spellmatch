# Installation

Spellmatch is implemented in Python and shipped as a Python package.

## Requirements

The spellmatch Python package requires Python 3.13 or newer.

Using virtual environments (e.g. created with [uv](https://docs.astral.sh/uv/)) is
strongly recommended for reproducibility reasons.

## Installation

You can install spellmatch [from PyPI](https://pypi.org/project/spellmatch/) using
[pip](https://pypi.org/project/pip/):

    pip install spellmatch

The latest development version can be installed
[directly from GitHub](https://github.com/BodenmillerGroup/spellmatch):

    pip install git+https://github.com/BodenmillerGroup/spellmatch

## Optional dependencies

By default, only the dependencies required for headless use (e.g. on a cluster) are
installed. Additional functionality can be enabled by installing optional dependencies
("extras"):

| Extra     | Functionality                                                        |
| --------- | -------------------------------------------------------------------- |
| `gui`     | Interactive cell matching (napari) and live registration progress    |
| `probreg` | Point set registration algorithms from probreg (CPD, FilterReg, ...) |
| `all`     | All of the above                                                     |
| `cuda`    | GPU acceleration for probreg algorithms (requires CUDA 12)           |
| `nonfree` | Patented OpenCV features (e.g. SURF) for feature-based registration  |

For example, to install spellmatch with all optional dependencies:

    pip install "spellmatch[all]"

!!! note "Installing probreg"
    The probreg package is only available as source distribution and requires a C++
    compiler for installation.

!!! warning "Installing nonfree"
    The `nonfree` extra installs `opencv-contrib-python-headless`, which provides the
    same `cv2` module as the `opencv-python-headless` package installed by default.
    After installing it, run
    `pip install --force-reinstall --no-deps opencv-contrib-python-headless` to ensure
    that the contrib version of the `cv2` module is used.

## Reproducing the publication

The analyses in `data/` were run with the original release of spellmatch (Python 3.9);
the corresponding conda environment is provided in `environment.yml`.
