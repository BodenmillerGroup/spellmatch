"""Matching algorithms based on point set registration algorithms adapted from probreg.

The adapted probreg code and its license are in ``_vendor``.
"""

from ._probreg import (
    TPSSVR,
    AffineCoherentPointDrift,
    CombinedBayesianCoherentPointDrift,
    NonRigidCoherentPointDrift,
    RigidCoherentPointDrift,
    RigidGMMReg,
    RigidSVR,
    TPSGMMReg,
    spellmatch_get_mask_matching_algorithm,
)

__all__ = [
    "AffineCoherentPointDrift",
    "CombinedBayesianCoherentPointDrift",
    "NonRigidCoherentPointDrift",
    "RigidCoherentPointDrift",
    "RigidGMMReg",
    "RigidSVR",
    "TPSGMMReg",
    "TPSSVR",
    "spellmatch_get_mask_matching_algorithm",
]
