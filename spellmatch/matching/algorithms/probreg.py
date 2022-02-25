from typing import Optional, Type

from ..._spellmatch import hookimpl
from ._algorithms import MaskMatchingAlgorithm, PointsMatchingAlgorithm

# TODO implement probreg algorithms


@hookimpl
def spellmatch_get_mask_matching_algorithm(
    name: str,
) -> Optional[Type[MaskMatchingAlgorithm]]:
    if name == "rigid_cpd":
        return RigidCoherentPointDrift
    if name == "affine_cpd":
        return AffineCoherentPointDrift
    if name == "nonrigid_cpd":
        return NonRigidCoherentPointDrift
    if name == "combined_bayesian_cpd":
        return CombinedBayesianCoherentPointDrift
    if name == "rigid_filterreg":
        return RigidFilterReg
    if name == "deformable_kinematic_filterreg":
        return DeformableKinematicFilterReg
    if name == "rigid_gmmreg":
        return RigidGMMReg
    if name == "tps_gmmreg":
        return TPSGMMReg
    if name == "rigid_svr":
        return RigidSVR
    if name == "tps_svr":
        return TPSSVR
    if name == "gmmtree":
        return GMMTree
    return None


class _Probreg(PointsMatchingAlgorithm):
    pass


class _CoherentPointDrift(_Probreg):
    pass


class RigidCoherentPointDrift(_CoherentPointDrift):
    pass


class AffineCoherentPointDrift(_CoherentPointDrift):
    pass


class NonRigidCoherentPointDrift(_CoherentPointDrift):
    pass


class _BayesianCoherentPointDrift(_Probreg):
    pass


class CombinedBayesianCoherentPointDrift(_BayesianCoherentPointDrift):
    pass


class _FilterReg(_Probreg):
    pass


class RigidFilterReg(_FilterReg):
    pass


class DeformableKinematicFilterReg(_FilterReg):
    pass


class _L2DistReg(_Probreg):
    pass


class RigidGMMReg(_L2DistReg):
    pass


class TPSGMMReg(_L2DistReg):
    pass


class RigidSVR(_L2DistReg):
    pass


class TPSSVR(_L2DistReg):
    pass


class GMMTree(_Probreg):
    pass
