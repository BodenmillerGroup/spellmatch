# Spatial cell matching

!!! warning "Under construction"
    This page is still under construction.

Spatial cell matching produces cell alignment scores using one of the
[matching algorithms](../algorithms/index.md). Algorithm parameters are specified as
comma-separated key-value pairs using the `--algorithm-args` option, for example:

    spellmatch match --algorithm spellmatch --algorithm-args "adj_radius=15,alpha=0.8,degree_weight=1" ...

::: mkdocs-click
    :module: spellmatch.__main__
    :command: cli_match
    :prog_name: spellmatch match
    :depth: 1
    :style: table
