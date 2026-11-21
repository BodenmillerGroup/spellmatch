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

For example, to install spellmatch with all optional dependencies:

    pip install "spellmatch[all]"

!!! note "Installing probreg"
    The probreg package is only available as source distribution and requires a C++
    compiler for installation.

!!! note "Feature-based registration"
    SIFT and ORB features are included in the default OpenCV installation. SURF is
    patented and excluded from the OpenCV packages on PyPI; it is only available with
    an OpenCV build with `OPENCV_ENABLE_NONFREE` enabled.

## Reproducing the publication

The analyses in `data/` (notebooks and benchmark scripts) require the optional GUI
dependencies, [simutome](https://github.com/BodenmillerGroup/simutome), and a few
analysis packages (JupyterLab, matplotlib, seaborn, datashader). From a clone of the
[spellmatch repository](https://github.com/BodenmillerGroup/spellmatch), all of them
can be installed with [uv](https://docs.astral.sh/uv/), using the exact versions
recorded in `uv.lock`:

    uv sync --extra all --group benchmark --group analysis
    uv run jupyter lab

Cell segmentation using Mesmer (`deepcell`), performed in some of the dataset notebooks,
requires a separate environment.
