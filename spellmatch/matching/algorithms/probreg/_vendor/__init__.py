"""Point set registration algorithms adapted from probreg.

Adapted from probreg 0.3.8 (https://github.com/neka-nat/probreg, MIT License,
Copyright (c) 2019 neka-nat, see LICENSE in this directory) as pure Python code, with
the following changes:

- Compiled extensions replaced by numpy/scipy implementations (kernels, exact Gauss
  transform instead of the improved fast Gauss transform), no open3d/transforms3d
- Arbitrary point dimensionality (probreg 0.3.8 hard-codes 3D point clouds in CPD and
  in the rigid L2 distance cost function)
- Rigid CPD: corrected variance update without scale estimation
- Rigid GMMReg/SVR: rotation parametrized by an angle in 2D
- CuPy kernels (CPD): dimension-independent implementations
- Combined BCPD: revised following Hirose (2021), see ``bcpd.py``
- FilterReg, GMMTree, colored CPD and constrained non-rigid CPD are not included
"""
