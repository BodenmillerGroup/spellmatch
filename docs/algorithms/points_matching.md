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
| `scores_tol` | | Convergence tolerance for the relative change in the cell alignment scores between iterations (Frobenius norm of the change, divided by the Frobenius norm of the current scores) |
| `transform_tol` | | Convergence tolerance (Frobenius norm) for changes in the spatial transform matrix between iterations |
| `max_dist` | | Maximum distance between matched (closest) cells |
| `filter_outliers` | `True` | Whether to exclude cells outside of the overlapping image region (bounding boxes, extended by `max_dist`) from matching; only effective if `max_dist` is specified |
| `min_change` | | Convergence tolerance for relative changes in the mean and standard deviation of distances between matched cells |

## Algorithms adapted from probreg

The following algorithms are implemented using
[probreg](https://github.com/neka-nat/probreg) and require the optional `probreg`
dependency (`pip install "spellmatch[probreg]"`). After registering the point sets,
each source cell is matched to its closest target cell.

All algorithms adapted from probreg share the following parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `outlier_dist` | | If specified, exclude cells outside of the overlapping image region (bounding boxes, extended by `outlier_dist`) from matching |
| `max_dist` | | Maximum distance between matched (closest) cells after registration |

Algorithm-specific parameters are passed to the corresponding probreg functions; please
refer to the [probreg documentation](https://probreg.readthedocs.io) for details.

| Algorithm | Name | Reference | Specific parameters (defaults) |
| --- | --- | --- | --- |
| Coherent Point Drift (CPD), rigid | `rigid_cpd` | [Myronenko and Song, 2010](https://doi.org/10.1109/tpami.2010.46) | `w` (`0`), `maxiter` (`50`), `tol` (`0.001`), `update_scale` (`True`) |
| CPD, affine | `affine_cpd` | | `w` (`0`), `maxiter` (`50`), `tol` (`0.001`) |
| CPD, non-rigid | `nonrigid_cpd` | | `w` (`0`), `maxiter` (`50`), `tol` (`0.001`), `beta` (`2`), `lmd` (`2`) |
| Bayesian Coherent Point Drift (BCPD) | `combined_bcpd` | [Hirose, 2021](https://doi.org/10.1109/tpami.2020.2971687) | `w` (`0`), `maxiter` (`50`), `tol` (`0.001`), `lmd` (`2`), `k` (`1e20`), `gamma` (`1`) |
| Gaussian Filter Registration (FilterReg), rigid | `rigid_filterreg` | [Gao and Tedrake, 2018](https://arxiv.org/abs/1811.10136) | `w` (`0`), `maxiter` (`50`), `tol` (`0.001`), `min_sigma2` (`0.0001`), `sigma2`, `update_sigma2` (`False`) |
| FilterReg, deformable kinematic | `deformable_kinematic_filterreg` | | `w` (`0`), `maxiter` (`50`), `tol` (`0.001`), `min_sigma2` (`0.0001`), `sigma2` |
| Gaussian Mixture Models Registration (GMMReg), rigid | `rigid_gmmreg` | [Jian and Vemuri, 2005](https://doi.org/10.1109/iccv.2005.17) | `maxiter` (`1`), `tol` (`0.001`), `opt_maxiter` (`50`), `opt_tol` (`0.001`), `sigma` (`1`), `delta` (`0.9`), `n_gmm_components` (`800`), `use_estimated_sigma` (`True`) |
| GMMReg, thin plate spline (TPS) | `tps_gmmreg` | | as `rigid_gmmreg`, plus `alpha` (`1`), `beta` (`0.1`) |
| Support Vector Registration (SVR), rigid | `rigid_svr` | [Campbell and Petersson, 2015](https://doi.org/10.1109/iccv.2015.488) | `maxiter` (`1`), `tol` (`0.001`), `opt_maxiter` (`50`), `opt_tol` (`0.001`), `sigma` (`1`), `delta` (`0.9`), `gamma` (`0.5`), `nu` (`0.1`), `use_estimated_sigma` (`True`) |
| SVR, thin plate spline (TPS) | `tps_svr` | | as `rigid_svr`, plus `alpha` (`1`), `beta` (`0.1`) |
| Gaussian Mixture Model Trees Registration (GMMTree) | `gmmtree` | [Eckart et al., 2018](https://arxiv.org/abs/1807.02587) | `tree_level` (`2`), `lambda_c` (`0.01`), `lambda_s` (`0.001`), `maxiter` (`20`), `tol` (`0.0001`) |

!!! note "GPU acceleration"
    If [CuPy](https://cupy.dev) is installed (`pip install "spellmatch[cuda]"`) and a
    CUDA device is available, the CPD algorithms run on the GPU.
