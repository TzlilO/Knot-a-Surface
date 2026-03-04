"""
Control feature modules for B-spline surface properties.

Each module wraps a learnable [H, W, C] control grid that is interpolated
via shared B-spline basis functions to produce [Us, Vs, C] sample values.
"""

from .base import ControlFeature
from .position import PositionControl
from .rotation import RotationControl
from .scaling import ScalingControl
from .opacity import OpacityControl
from .weights import WeightControl
from .sh import SHControl, SHControlWrapper
from .quaternion_utils import quaternion_mean, slerp, batch_slerp, matrix_to_quaternion

__all__ = [
    'ControlFeature',
    'PositionControl',
    'RotationControl',
    'ScalingControl',
    'OpacityControl',
    'WeightControl',
    'SHControl',
    'SHControlWrapper',
    'quaternion_mean',
    'slerp',
    'batch_slerp',
    'matrix_to_quaternion',
]