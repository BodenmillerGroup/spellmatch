# Introduction

In this chapter, the usage of spellmatch is introduced by example. The dataset being
showcased is a subset of 
[(Kuett and Catena et al., 2022)](https://doi.org/10.5281/zenodo.4752030), which can be
found in `data/kuett_catena_2022` of the
[spellmatch repository](https://github.com/BodenmillerGroup/spellmatch).

## Workflow

The following flowchart illustrates the spellmatch workflow. Blue rectangles indicate
*steps*, yellow boxes represent *modules*. Cylinders indicate input and output data of
the workflow.

```mermaid
flowchart TB
    source_data[("Source data<br>(masks, images, panel)")] --> initial_registration
    target_data[("Target data<br>(masks, images, panel)")] --> initial_registration

    subgraph Registration
        initial_registration["Interactive cell matching<br>OR<br>Feature-based image registration"]
        refined_registration[Intensity-based image registration]

        initial_registration -->|"initial transform"| refined_registration
    end

    initial_registration -->|"initial assignment<br>(for validation, optional)"| assignment
    refined_registration -->|refined transform| matching

    subgraph Matching
        matching[Automatic cell matching]
        transform_update["Transform estimation<br>(iterative algorithms only)"]        

        matching -->|matching scores| transform_update
        transform_update -->|updated transform| matching
    end

    matching -->|matching scores| assignment

    subgraph Assignment
        assignment["Cell assignment<br>(directed/undirected)"]
        combination["Assignment combination<br>(directed assignments only)"]

        assignment -->|"forward assignment<br>reverse assignment"| combination
    end
    
    combination --> cell_pairs[("Cell assignment<br>(directed/undirected)")]
    transform_update --> transform[(Geometric transform)]
```

## Usage

Spellmatch can be used from the command-line (terminal in Linux/MacOS, console in
Windows):

    ❯ spellmatch --help
    Usage: spellmatch [OPTIONS] COMMAND [ARGS]...

    Options:
    -v, --verbosity LVL  Either CRITICAL, ERROR, WARNING, INFO or DEBUG
    --version            Show the version and exit.
    --help               Show this message and exit.

    Commands:
    ...

As indicated above, all spellmatch commands support `-v` option for enabling more
verbose output. In addition, many spellmatch commands support a `--show` option for
graphical visualization of the current operation. At any point, use the `--help` option
to display additional information about a specific command.

All spellmatch commands can operate on individual file pairs as well as on entire
directories.

## Input

Spellmatch requires pairs of source data and target data, where source/target data
consist of:

- 2D cell masks (TIFF files of any data type)
- 2D single- or multichannel images (TIFF files of any data type, optional)
- For multichannel images: panel with channel information (CSV files, in channel order)  
  Column headers: `name` (channel name, unique), `keep` (`0` or `1`, optional)

Images and corresponding masks have to match in size. For multichannel images, the
number of rows in the panel (exluding column headers) has to match the number of image
channels. Images and masks are matched by filename (alphabetical order).

Source and target data are matched by filename (alphabetical order). Source and target
images/masks do not have to have the same size, scale (pixel size), or number of
channels. Source panel and target panel can share channels, but do not have to.

## Tasks

For convenience, multiple steps are combined into *tasks*. For each task, spellmatch
provides a dedicated *command*. The individual tasks and commands are described on the
following pages.

1. [Image registration](registration.md)
    - Interactive cell matching
    - Feature-based image registration
    - Intensity-based image registration
2. [Spatial cell matching](matching.md)
    - Automatic cell matching
    - Transform estimation (iterative algorithms only)
3. [Cell assignment](assignment.md)
    - Cell assignment (directed/undirected)
    - Assignment combination (directed assignments only)

## Output

Spellmatch produces the following output data:

- Projective transformations (3x3 numpy array, stored as numpy `.npy` files)  
  *Note: geometric transforms are computed from centered masks/images*
- Matching scores (xarray DataArray, stored as netCDF `.nc` files)  
  Shape: source labels x target labels, data type: floating point
- Directed/undirected assignments (CSV files holding cell label pairs)  
  Column headers: `Source` (source cell label), `Target` (target cell label)

By default, all files are named `{source}_to_{target}.{suffix}`, where `{source}`
corresponds to the source mask/image name and `{target}` corresponds to the target
mask/image name.