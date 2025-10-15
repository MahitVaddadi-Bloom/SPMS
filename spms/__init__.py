"""
SPMS: Spherical Projection Molecular Surface descriptors.

A package for generating molecular descriptors from 3D conformers using 
spherical projection methodology.
"""

__version__ = "1.0.0"

from spms.desc import SPMS
from spms import numpy_compat
from spms import cli

__all__ = ["SPMS", "numpy_compat", "cli"]