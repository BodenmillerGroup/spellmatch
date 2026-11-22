# Image registration

!!! warning "Under construction"
    This page is still under construction.

Image registration estimates the spatial alignment (projective transform) between source
and target images, which can be used as a spatial prior for cell matching.

The following registration methods are available:

- **Interactive cell matching**: cells are matched manually in
  [napari](https://napari.org). The spatial transform is estimated from the matched
  cells, and the matched cells can be used for validation (requires the optional `gui`
  dependency: `pip install "spellmatch[gui]"`).
- **Feature-based image registration**: the spatial transform is estimated from matched
  image features (e.g. SIFT, ORB) using [OpenCV](https://opencv.org).
- **Intensity-based image registration**: the spatial transform is estimated by
  optimizing an image similarity metric using [SimpleITK](https://simpleitk.org).

::: mkdocs-click
    :module: spellmatch.__main__
    :command: cli_register
    :prog_name: spellmatch register
    :depth: 1
    :style: table
