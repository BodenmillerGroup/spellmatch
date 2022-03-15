# Graph matching

!!! warning "Under construction"
    This page is still under construction.

<!-- TODO explain graph construction -->

## Spellmatch

Windhager and Bodenmiller, 2022 (in preparation)

The *Spellmatch* algorithm is based on the *FINAL* algorithm for attributed network
alignment ([Zhang and Tong, 2018](https://doi.org/10.1109/TKDE.2018.2866440)). It
operates under the assumption that the tissue topology as given by spatial cell graphs
is preserved across neighboring tissue sections. In addition to tissue topology, it
considers cell features (number of neighbors, intensities) and spatial cell-cell
distances to yield a globally optimal probabilistic matching. Notably, the *Spellmatch*
algorithm adopts the *MARIO* 
([Zhu et al., 2021](https://doi.org/10.1101/2021.12.03.471185)) strategy for balancing
similarity scores computed from cell intensity information of shared (intersection) and
combined (union) markers.

The *Spellmatch* algorithm is an *iterative* algorithm. As such, it can take the initial
spatial alignment (cf. image co-registration) as a weighted prior for matching cells,
while still allowing for matches that may not be captured by the prior. Furthermore, the
*Spellmatch* algorithm can be *constrained* to only match cells in spatial proximity.

!!! note "The *Spellmatch* algorithm and the Iterative Closest Points (ICP) algorithm"
    By setting all weights to `0`, or by setting `alpha` to `0`, the *Spellmatch*
    algorithm approximately converges to the Iterative Closest Points (ICP) algorithm.

!!! note "The *Spellmatch* algorithm and the Generalized Assignment Problem (GAP)"
    The `adj_radius` parameter controls the neighborhood radius of a cell. In theory, by
    setting this parameter to infinity (or to a value larger than the diagonal of the 
    larger image), the *Spellmatch* algorithm converges to a solver for the
    [generalized assignment problem (GAP)](https://en.wikipedia.org/wiki/Generalized_assignment_problem).
    In other words, the *Spellmatch* algorithm can be seen as a GAP solver, with
    a search space constrained to the local neighborhood of cells.

| Parameter | Default value | Description |
| --- | --- | --- |
| `point_feature` | `centroid` | [scikit-image region property](https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops) for spatial coordinates |
| `intensity_feature` | `intensity_mean` | [scikit-image region property](https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops) for cell intensities |
| `intensity_transform` | | Intensity transform function, e.g. `numpy.arcsinh` |
| `transform_type` | `rigid` | Spatial transform model (`rigid`, `similarity`, `affine`) |
| `transform_estim_type` | `max_score` | Transform estimation strategy (`max_score`, `max_margin`) |
| `transform_estim_k_best` | `50` | Number of points to use for estimating transforms |
| `max_iter` | `10` | Number of iterations (transform estimations) |
| `scores_tol` | | Frobenius tolerance for changes in scores matrix |
| `transform_tol` | | Frobenius tolerance for changes in transform matrix |
| `filter_outliers` | `True` | Whether to include outliers (*strongly recommended*) |
| `adj_radius` | `15` | Physical radius in which to consider cells neighbors |
| `alpha` | `0.8` | Spatial prior importance [0..very important, 1..ignore] |
| `degree_weight`<br>`intensity_weight`<br>`celldist_weight` | `0`<br>`0`<br>`0` | Relative contributions (weights) of cell degree<br> (number of neighbors), cell intensity, and<br>cell-cell distance |
| `degree_cdiff_thres` | `3` | Degree cross-difference threshold |
| `shared_intensity...`<br>`pca_n_components` | <br>`5` | Principal components analysis (PCA) configuration for<br>computing the shared intensity feature cross-distance |
| `full_intensity_...`<br>`cca_fit_k_closest`<br>`fit_k_most_certain`<br>`cca_n_components` | <br>`500`<br>`100`<br>`10` | Canonical Correlation Analysis (CCA) configuration for<br>computing the full intensity feature cross-distance |
| `intensity_interp_...`<br>`lmd`<br>`cca_n_components` | <br>`11`<br>`10` | Parameters for shared/full intensity feature<br> cross-distance interpolation (weighting) |
| `spatial_cdist_`<br>`prior_thres` | | Distance threshold for the spatial cross-distance prior |
| `max_spatial_cdist` | | Constraint for only matching cells in spatial proximity |
| `cca_max_iter`<br>`cca_tol` | `500`<br>`1.0e-6` | Shared Canonical Correlation Analysis (CCA) parameters |
| `opt_max_iter`<br>`opt_tol` | `100`<br>`1.0e-9` | Parameters for the iterative optimization scheme |
