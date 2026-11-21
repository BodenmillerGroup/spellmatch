# Graph matching

!!! warning "Under construction"
    This page is still under construction.

<!-- TODO explain graph construction -->

## Spellmatch

Windhager and Bodenmiller (in preparation)

Algorithm name: `spellmatch`

The *Spellmatch* algorithm is based on the *FINAL* algorithm for attributed network
alignment ([Zhang and Tong, 2018](https://doi.org/10.1109/TKDE.2018.2866440)). It
operates under the assumption that the tissue topology as given by spatial cell graphs
is preserved across neighboring tissue sections. In addition to tissue topology, it
considers cell features (number of neighbors, intensities) and spatial cell-cell
distances to yield a globally optimal matching. Notably, the *Spellmatch* algorithm
adopts the *MARIO* ([Zhu et al., 2021](https://doi.org/10.1101/2021.12.03.471185))
strategy for balancing similarity scores computed from cell intensity information of
shared (intersection) and combined (union) markers.

The *Spellmatch* algorithm is an *iterative* algorithm. As such, it can take the initial
spatial alignment (cf. image co-registration) as a weighted prior for matching cells
(`spatial_cdist_prior_thres`), while still allowing for matches that may not be captured
by the prior. Furthermore, the *Spellmatch* algorithm can be *constrained* to only match
cells in spatial proximity (`max_spatial_cdist`): cell pairs farther apart under the
current spatial transform are excluded from the optimization and receive a score of
zero.

!!! note "The *Spellmatch* algorithm and the Iterative Closest Points (ICP) algorithm"
    By setting `alpha` to `0` (and specifying `spatial_cdist_prior_thres`), the cell
    alignment scores are determined by the spatial prior alone. In this case, the
    *Spellmatch* algorithm behaves similarly to the Iterative Closest Points (ICP)
    algorithm. Note that setting all weights (`degree_weight`, `intensity_weight`,
    `distance_weight`) to `0` does not have this effect: in this case, cells are matched
    based on tissue topology (spatial cell graphs) only. This is equivalent to the
    attribute-free variant of *FINAL* (*FINAL-P*), i.e., an *IsoRank*-like alignment
    ([Singh et al., 2008](https://doi.org/10.1073/pnas.0806627105)).

!!! note "The *Spellmatch* algorithm and the Generalized Assignment Problem (GAP)"
    The `adj_radius` parameter controls the neighborhood radius of a cell. In theory, by
    setting this parameter to infinity (or to a value larger than the diagonal of the
    larger image), the *Spellmatch* algorithm converges to a solver for the
    [generalized assignment problem (GAP)](https://en.wikipedia.org/wiki/Generalized_assignment_problem).
    In other words, the *Spellmatch* algorithm can be seen as a GAP solver, with
    a search space constrained to the local neighborhood of cells.

Parameters (symbols refer to the manuscript, where applicable):

| Parameter | Symbol | Default | Description |
| --- | --- | --- | --- |
| `point_feature` | | `centroid` | [scikit-image region property](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops) used as spatial cell coordinates |
| `intensity_feature` | | `intensity_mean` | [scikit-image region property](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops) used as cell intensities |
| `intensity_transform` | | | Intensity transform function, e.g. `numpy.arcsinh` |
| `transform_type` | | `rigid` | Spatial transform model (`rigid`, `similarity`, `affine`) |
| `transform_estim_type` | | `max_score` | Transform estimation strategy (`max_score`, `max_margin`) |
| `transform_estim_k_best` | n<sub>𝒯</sub> | `50` | Number of top-matched cells used for estimating the spatial transform |
| `max_iter` | n<sub>maxiter</sub> | `50` | Maximum number of iterations (spatial transform estimations) |
| `scores_tol` | ε<sub>RGA</sub> | | Convergence tolerance (Frobenius norm) for changes in the cell alignment scores between iterations; if not specified, `max_iter` iterations are performed |
| `transform_tol` | | | Convergence tolerance (Frobenius norm) for changes in the spatial transform matrix between iterations |
| `filter_outliers` | | `True` | Whether to exclude cells outside of the overlapping image region (bounding boxes, extended by `adj_radius`) from matching |
| `adj_radius` | r<sub>adj</sub> | `15` | Physical radius in which cells are considered neighbors |
| `alpha` | α | `0.8` | Regularization parameter in `[0, 1)` (`0`: spatial prior only; the closer to `1`, the smaller the influence of the spatial prior) |
| `spatial_cdist_prior_thres` | θ<sub>prior</sub> | | Spatial cell cross-distance threshold for the spatial prior (soft constraint); if not specified, a uniform prior is used |
| `max_spatial_cdist` | d<sub>max</sub> | | Maximum spatial cell cross-distance for matching cells (hard constraint); if not specified, all cell pairs are considered |
| `degree_weight` | w<sub>degree</sub> | `0` | Relative importance of cell degrees (number of neighbors) |
| `degree_cdiff_thres` | θ<sub>degree</sub> | `3` | Cell degree difference threshold |
| `intensity_weight` | w<sub>intensity</sub> | `0` | Relative importance of cell intensities |
| `intensity_interp_lmd` | λ, n<sup>(try)</sup><sub>λ-sel</sub> | `11` | Weight of the shared-marker intensity cross-distance relative to the all-marker one: a number in `[0, 1]` specifies λ; an integer larger than `1` specifies the number of regularly spaced λ values to evaluate for automatic selection (the selected λ is reported in the scores info; `None` if intensities are not used) |
| `intensity_interp_cca_n_components` | n<sup>(cancor)</sup><sub>λ-sel</sub> | `10` | Number of canonical correlations used for automatic λ selection |
| `intensity_shared_pca_n_components` | k<sub>SVD</sub> | `5` | Number of SVD components for the shared-marker intensity cross-distance. Shared-marker intensities are standardized per dataset; markers without variance in either dataset are excluded. With only one (two) shared marker(s), the SVD is skipped; with only one (two) shared marker(s) or SVD component(s), the specified metric is replaced by the squared difference relative to `intensity_1d_cdiff_thres_sd` (pooled) standard deviations, clamped to [0, 1] (cosine distance) |
| `intensity_shared_metric` | | `reconstructed_correlation` | Distance between SVD-embedded shared-marker intensities: `reconstructed_correlation` (Pearson correlation across markers between rank-k SVD reconstructions, as in MARIO), `cosine` (cosine distance between SVD embeddings), or `correlation` (Pearson correlation across SVD components; default of spellmatch 0.1) |
| `intensity_all_cca_fit_k_closest` | n<sup>(closest)</sup><sub>CCA</sub> | `500` | Number of spatially closest cell pairs considered for fitting the CCA (all-marker intensity cross-distance); limited to the number of source cells |
| `intensity_all_cca_fit_k_most_certain` | n<sup>(most certain)</sup><sub>CCA</sub> | `100` | Number of most certain cell pairs (largest spatial margins) used for fitting the CCA; limited to n<sup>(closest)</sup><sub>CCA</sub> |
| `intensity_all_cca_n_components` | k<sub>CCA</sub> | `10` | Number of CCA components for the all-marker intensity cross-distance; limited to the number of cells used for fitting the CCA. With only one (two) CCA component(s), the specified metric is replaced by the squared difference relative to `intensity_1d_cdiff_thres_sd` (pooled) standard deviations, clamped to [0, 1] (cosine distance) |
| `intensity_all_metric` | | `cosine` | Distance between CCA-embedded intensities: `cosine` (cosine distance between CCA embeddings) or `correlation` (Pearson correlation across CCA components; default of spellmatch 0.1) |
| `intensity_1d_cdiff_thres_sd` | | `2` | Intensity difference threshold, in (pooled) standard deviations, for one-dimensional intensity cross-distances (only one shared marker, SVD component, or CCA component) |
| `distance_weight` | w<sub>distance</sub> | `0` | Relative importance of spatial cell-cell distances |
| `distance_cdiff_thres` | θ<sub>distance</sub> | `5` | Spatial cell-cell distance difference threshold |
| `cca_max_iter`<br>`cca_tol` | | `500`<br>`1e-6` | Canonical correlation analysis (CCA) fitting parameters |
| `opt_max_iter`<br>`opt_tol` | n<sub>maxopt</sub><br>ε<sub>FPI</sub> | `200`<br>`1e-6` | Maximum number of iterations and convergence tolerance of the fixed point iteration |
| `precision` | | `float32` | Floating point precision used for computations |
