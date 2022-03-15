# Points matching

!!! warning "Under construction"
    This page is still under construction.

<!-- TODO explain points extraction -->

## Iterative Closest Points (ICP)

[Besl and McKay, 1992](https://doi.org/10.1109/34.121791)

Implemented based on [simpleICP](https://github.com/pglira/simpleICP)

## Coherent Point Drift (CPD)

[Myronenko and Song, 2009](https://arxiv.org/abs/0905.2635)

Implemented using [probreg](https://github.com/neka-nat/probreg)

Variants: rigid, affine, non-rigid

## Gaussian Filter Registration (FilterReg)

[Gao and Tedrake, 2019](https://arxiv.org/abs/1811.10136)

Implemented using [probreg](https://github.com/neka-nat/probreg)

Variants: rigid, deformable kinematic

## Bayesian Coherent Point Drift (BCPD)

[Hirose, 2021](https://doi.org/10.1109/TPAMI.2020.2971687)

Implemented using [probreg](https://github.com/neka-nat/probreg)

## Gaussian Mixture Models Registration (GMMReg)

[Jian and Vemuri, 2010](https://doi.org/10.1109/TPAMI.2010.223)

Implemented using [probreg](https://github.com/neka-nat/probreg)

Variants: rigid, TPS

## Support Vector Registration (SVR)

[Campbell and Petersson, 2015](https://arxiv.org/abs/1511.04240)

Implemented using [probreg](https://github.com/neka-nat/probreg)

Variants: rigid, TPS

## Gaussian Mixture Model Trees Registration (GMMTree)

[Eckart et al., 2018](https://arxiv.org/abs/1807.02587)

Implemented using [probreg](https://github.com/neka-nat/probreg)
