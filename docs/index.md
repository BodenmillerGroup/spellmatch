# Welcome

!!! warning "Under development"
    This tool is still under active development.

Spellmatch is a command-line tool for spatial cell representation matching in segmented
2D images. Its main use case is multi-panel imaging, where consecutive tissue slices are
imaged with partially overlapping panels, e.g. to enhance multiplexity.

!!! note "Spellmatch matches spatial cell representations"
    Unlike other approaches, spellmatch is not meant for matching similar cells, but
    matches *different representations of the same (i.e., identical) cell* instead.
    Importantly, spellmatch only includes algorithms for *spatial* cell matching, i.e.
    algorithms that consider the spatial distributions of cells in tissue.

## Overview

Spellmatch provides the following functionality:

- Automatic feature/intensity-based image registration
- Interactive cell matching for manual image registration and validation
- Point set registration for matching cells based on their spatial distribution
- Network alignment for matching cells based on their topology and attributes (e.g.
  intensities)
- Extraction of directed and/or undirected cell-cell assignments from matching scores

Notably, spellmatch ships with its own network alignment algorithm, which leverages the
full information present in multichannel images to obtain a probabilistic cell matching.
For an overview of all matching algorithms implemented in spellmatch, please refer to
[Algorithms](algorithms/index.md).

For developers, spellmatch can also be used as a Python package, e.g. to extend
spellmatch or to integrate it into custom pipelines. Please refer to
[Reference](reference/index.md) for a detailed API documentation

## Experiments

To ensure spellmatch yields high-quality results, your dataset needs to meet certain
experimental requirements. Please refer to [Protocol](protocol/index.md) for
instructions related to data acquisition protocols.

## Getting started

[Install spellmatch](installation.md)

[Use spellmatch](usage/index.md)

## Getting help

A preprint is in preparation.

For conceptual questions, please do not hesitate to
[reach out](mailto:jonas.windhager@uzh.ch).

For technical issues, please refer to the
[spellmatch issue tracker](https://github.com/BodenmillerGroup/spellmatch/issues).

## Citing spellmatch

A preprint is in preparation.

If you are using spellmatch in your work, please
[reach out](mailto:jonas.windhager@uzh.ch) to inquire how to best cite the tool.
