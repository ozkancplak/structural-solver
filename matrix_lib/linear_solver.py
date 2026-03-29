"""
linear_solver.py
================
Purpose : Factory interface that selects the appropriate matrix storage
          scheme and solver for a given system  A * x = b.

Units   : dimensionless  (caller is responsible for consistent units)
Inputs  : (see LinearSolver.solve docstring)
Outputs : solution vector x (list[float])

Notes
-----
* numpy is NOT used anywhere in this module.
* Available methods: 'dense', 'banded', 'skyline'  (default: 'skyline')
"""

from .dense_matrix   import DenseMatrix
from .banded_matrix  import BandedMatrix
from .skyline_matrix import SkylineMatrix


class LinearSolver:
    """
    Factory class for solving symmetric positive-definite linear systems.

    Attributes
    ----------
    method : str  – storage/solver type ('dense', 'banded', 'skyline')

    Methods
    -------
    solve(K_matrix, b, **kwargs)  – solve K * x = b
    """

    SUPPORTED_METHODS = ('dense', 'banded', 'skyline')

    def __init__(self, method: str = 'skyline'):
        """
        Inputs  : method (str) – one of 'dense', 'banded', 'skyline'
        Outputs : LinearSolver instance
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"LinearSolver: unknown method '{method}'. "
                f"Choose from {self.SUPPORTED_METHODS}."
            )
        self.method = method

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def solve(self, K_matrix, b: list, **kwargs) -> list:
        """
        Solve  K * x = b  using the configured storage scheme.

        Parameters
        ----------
        K_matrix : one of DenseMatrix | BandedMatrix | SkylineMatrix
                   OR a list-of-lists (auto-converted to DenseMatrix)
        b        : list[float] – right-hand-side vector
        kwargs   : extra arguments (e.g. bandwidth=m for 'banded')

        Returns
        -------
        x : list[float]

        Assumptions
        -----------
        * K must be symmetric positive-definite.
        * Caller must ensure K and b are consistent in size.
        """
        if isinstance(K_matrix, list):
            # Auto-convert plain list-of-lists to DenseMatrix
            n = len(K_matrix)
            dm = DenseMatrix(n, n)
            for i in range(n):
                for j in range(n):
                    dm[i, j] = K_matrix[i][j]
            return dm.solve(b)

        if isinstance(K_matrix, DenseMatrix):
            return K_matrix.solve(b)

        if isinstance(K_matrix, BandedMatrix):
            return K_matrix.solve(b)

        if isinstance(K_matrix, SkylineMatrix):
            return K_matrix.solve(b)

        raise TypeError(
            f"LinearSolver.solve: unsupported K_matrix type '{type(K_matrix).__name__}'."
        )

    # ------------------------------------------------------------------
    # Convenience class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def solve_dense(cls, K_list: list, b: list) -> list:
        """
        Convenience: solve using DenseMatrix (Gauss elimination).
        Inputs  : K_list (list-of-lists n×n), b (list[float])
        Outputs : x (list[float])
        """
        n = len(K_list)
        dm = DenseMatrix(n, n)
        for i in range(n):
            for j in range(n):
                dm[i, j] = K_list[i][j]
        return dm.solve(b)

    @classmethod
    def solve_banded(cls, K_banded: BandedMatrix, b: list) -> list:
        """
        Convenience: solve using an existing BandedMatrix.
        Inputs  : K_banded (BandedMatrix), b (list[float])
        Outputs : x (list[float])
        """
        return K_banded.solve(b)

    @classmethod
    def solve_skyline(cls, K_skyline: SkylineMatrix, b: list) -> list:
        """
        Convenience: solve using an existing SkylineMatrix.
        Inputs  : K_skyline (SkylineMatrix), b (list[float])
        Outputs : x (list[float])
        """
        return K_skyline.solve(b)

    def __repr__(self) -> str:
        return f"LinearSolver(method='{self.method}')"
