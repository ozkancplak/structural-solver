"""
matrix_lib
==========
CE 4011 Assignment #2 – Pure-Python OOP Matrix Library.

Public API
----------
from matrix_lib import DenseMatrix, BandedMatrix, SkylineMatrix, LinearSolver
"""

from .dense_matrix   import DenseMatrix
from .banded_matrix  import BandedMatrix
from .skyline_matrix import SkylineMatrix
from .linear_solver  import LinearSolver

__all__ = ["DenseMatrix", "BandedMatrix", "SkylineMatrix", "LinearSolver"]
