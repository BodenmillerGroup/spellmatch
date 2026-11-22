# Points matching

!!! warning "Under construction"
    This page is still under construction.

All points matching algorithms share the following parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `point_feature` | `centroid` | [scikit-image region property](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops) used as spatial cell coordinates |
| `intensity_feature` | `intensity_mean` | [scikit-image region property](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops) used as cell intensities |
| `intensity_transform` | | Intensity transform function, e.g. `numpy.arcsinh` |

## Iterative Closest Points (ICP)

[Besl and McKay, 1992](https://doi.org/10.1109/34.121791)

Algorithm name: `icp`

Implementation based on [simpleICP](https://github.com/pglira/simpleICP)

| Parameter | Default | Description |
| --- | --- | --- |
| `transform_type` | `rigid` | Spatial transform model (`rigid`, `similarity`, `affine`) |
| `max_iter` | `200` | Maximum number of iterations |
| `scores_tol` | `1e-6` | Convergence tolerance for the relative change in the cell alignment scores between iterations (Frobenius norm of the change, divided by the Frobenius norm of the current scores); the default stops once the assignments no longer change; if `None`, `max_iter` iterations are performed |
| `transform_tol` | | Convergence tolerance (Frobenius norm) for changes in the spatial transform matrix between iterations |
| `max_dist` | | Maximum distance between matched (closest) cells |
| `filter_outliers` | `True` | Whether to exclude cells outside of the overlapping image region (bounding boxes, extended by `max_dist`) from matching; only effective if `max_dist` is specified |
| `min_change` | | Convergence tolerance for relative changes in the mean and standard deviation of distances between matched cells |

## Algorithms adapted from probreg

The following algorithms are adapted from
[probreg](https://github.com/neka-nat/probreg) 0.3.8 (MIT License, Copyright (c) 2019
neka-nat) and are included in spellmatch as pure Python code. Compared to probreg, they
support 2D point sets and fix several bugs (see `spellmatch/matching/algorithms/_probreg`).
GMMReg, SVR and BCPD operate on normalized point sets (common centroid and standard
deviation), i.e., their parameters are relative to the extent of the point sets. After
registering the point sets, each source cell is matched to its closest target cell.

All algorithms adapted from probreg share the following parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `outlier_dist` | | If specified, exclude cells outside of the overlapping image region (bounding boxes, extended by `outlier_dist`) from matching |
| `max_dist` | | Maximum distance between matched (closest) cells after registration |

Algorithm-specific parameters are described in the
[probreg documentation](https://probreg.readthedocs.io) and the referenced publications.

| Algorithm | Name | Reference | Specific parameters (defaults) |
| --- | --- | --- | --- |
| Coherent Point Drift (CPD), rigid | `rigid_cpd` | [Myronenko and Song, 2010](https://doi.org/10.1109/tpami.2010.46) | `w` (`0`), `maxiter` (`50`), `tol` (`0.001`), `update_scale` (`True`) |
| CPD, affine | `affine_cpd` | | `w` (`0`), `maxiter` (`50`), `tol` (`0.001`) |
| CPD, non-rigid | `nonrigid_cpd` | | `w` (`0`), `maxiter` (`50`), `tol` (`0.001`), `beta` (`2`), `lmd` (`2`) |
| Bayesian Coherent Point Drift (BCPD) | `combined_bcpd` | [Hirose, 2021](https://doi.org/10.1109/tpami.2020.2971687) | `w` (`0`), `maxiter` (`50`), `tol` (`0.001`), `lmd` (`50`), `beta` (`2`), `k` (`1e20`), `gamma` (`1`) |
| Gaussian Mixture Models Registration (GMMReg), rigid | `rigid_gmmreg` | [Jian and Vemuri, 2005](https://doi.org/10.1109/iccv.2005.17) | `maxiter` (`1`), `tol` (`0.001`), `opt_maxiter` (`50`), `opt_tol` (`0.001`), `sigma` (`1`), `delta` (`0.9`), `n_gmm_components` (`800`), `use_estimated_sigma` (`True`) |
| GMMReg, thin plate spline (TPS) | `tps_gmmreg` | | as `rigid_gmmreg`, plus `alpha` (`1`), `beta` (`0.1`) |
| Support Vector Registration (SVR), rigid | `rigid_svr` | [Campbell and Petersson, 2015](https://doi.org/10.1109/iccv.2015.488) | `maxiter` (`1`), `tol` (`0.001`), `opt_maxiter` (`50`), `opt_tol` (`0.001`), `sigma` (`1`), `delta` (`0.9`), `gamma` (`0.5`), `nu` (`0.1`), `use_estimated_sigma` (`True`) |
| SVR, thin plate spline (TPS) | `tps_svr` | | as `rigid_svr`, plus `alpha` (`1`), `beta` (`10`) |

!!! note "GPU acceleration"
    If [CuPy](https://cupy.dev) is installed (`pip install "spellmatch[cuda]"`) and a
    CUDA device is available, the CPD algorithms run on the GPU (not tested in CI).

!!! note "Computational complexity"
    BCPD uses dense matrices of size (number of source cells)^2 and inverts one in each
    iteration, which is slow for more than a few thousand cells.
