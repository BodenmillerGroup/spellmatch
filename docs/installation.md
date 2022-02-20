# Installation

*Spellmatch* is implemented in Python and shipped as a Python package.

## Requirements

The `spellmatch` Python package requires Python 3.9 or newer.

A user interface for Jupyter notebooks, e.g.
[JupyterLab](https://jupyterlab.readthedocs.io) (optional, for opening notebooks).

Using virtual environments (e.g. [conda](https://docs.conda.io)) is strongly recommended
for reproducibility reasons.

!!! note "Hardware resources"
    Several spatial cell matching algorithms implemented in *spellmatch* require
    extensive hardware resources, depending on image sizes, hyper-parameters etc. A
    system with at least 64GB of RAM is recommended.

## Installation

You can install *spellmatch* [from PyPI](https://pypi.org/project/spellmatch/) using
[pip](https://pypi.org/project/pip/):

    pip install spellmatch

<!-- TODO PyPI -->

Alternatively, you can install *spellmatch*
[from conda-forge](https://github.com/conda-forge/spellmatch-feedstock) using
[conda](https://docs.conda.io/en/latest/):

    conda install -c conda-forge spellmatch

<!-- TODO conda-forge -->

## Usage

*Spellmatch* can be used from the command-line (terminal in Linux/MacOS, console in
Windows):

    > spellmatch --help
    Usage: spellmatch [OPTIONS] COMMAND [ARGS]...

    Options:
    --version  Show the version and exit.
    --help     Show this message and exit.

    Commands:
    ...

Refer to [Usage](usage/index.md) for detailed instructions on how to use *spellmatch*.
