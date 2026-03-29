"""
banded_matrix.py
================
Purpose : Symmetric banded matrix stored in compact column-major format.
          Implements LDLT (Crout) factorisation and forward/back
          substitution for efficient solution of banded linear systems.

Units   : dimensionless  (caller is responsible for consistent units)
Inputs  : (see individual method docstrings)
Outputs : (see individual method docstrings)

Storage layout
--------------
For a symmetric banded matrix of order n with half-bandwidth m (the
matrix has at most m+1 non-zero entries per row above and including
the diagonal), only the diagonal + upper m super-diagonals are stored.

  _data[i][k]  = A[i, i+k],   k = 0, …, min(m, n-1-i)

So row i has  min(m, n-1-i) + 1  entries.

Notes
-----
* numpy is NOT used anywhere in this module.
"""

import math


class BandedMatrix:
    """
    Symmetric positive-definite banded matrix with compact storage.

    Parameters
    ----------
    n         : int  – matrix order (n × n)
    bandwidth : int  – half-bandwidth  m  (number of super-diagonals)

    Attributes
    ----------
    n         : int
    bandwidth : int
    _data     : list[list[float]]  – compact upper-band storage
    _factored : bool  – True after LDLT factorisation
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, n: int, bandwidth: int):
        """
        Inputs  : n (int), bandwidth (int)
        Outputs : BandedMatrix initialised with zeros
        Assumptions : bandwidth >= 0,  bandwidth < n
        """
        self.n = n
        self.bandwidth = bandwidth
        self._factored = False

        # row i stores A[i, i], A[i, i+1], …, A[i, i+m]  (clipped at n-1)
        m = bandwidth
        self._data = [
            [0.0] * (min(m, n - 1 - i) + 1)
            for i in range(n)
        ]

    # ------------------------------------------------------------------
    # Element access
    # ------------------------------------------------------------------

    def __getitem__(self, key) -> float:
        """
        A[i, j]  – zero-indexed.  Only upper triangle / diagonal accessible.
        Inputs  : key = (i, j) tuple
        Outputs : float  (returns 0.0 for out-of-band entries; symmetry applied)
        """
        i, j = key
        # Enforce symmetry: always access upper triangle
        if j < i:
            i, j = j, i
        k = j - i
        if k > self.bandwidth or k >= len(self._data[i]):
            return 0.0
        return self._data[i][k]

    def __setitem__(self, key, value: float):
        """
        A[i, j] = value  – zero-indexed.
        Inputs  : key = (i, j) tuple, value (float)
        Outputs : None  (in-place)
        Raises  : ValueError for out-of-band assignments (non-zero)
        """
        i, j = key
        if j < i:
            i, j = j, i          # symmetry
        k = j - i
        if k > self.bandwidth:
            if value != 0.0:
                raise ValueError(
                    f"BandedMatrix: cannot set out-of-band entry "
                    f"A[{i},{j}] (half-bandwidth={self.bandwidth}) to {value}."
                )
            return
        self._data[i][k] = float(value)

    def add_to(self, i: int, j: int, value: float):
        """
        A[i, j] += value  (convenience accumulation method for assembly).
        Inputs  : i, j (int) – indices;  value (float)
        Outputs : None  (in-place)
        """
        if j < i:
            i, j = j, i
        k = j - i
        if k > self.bandwidth:
            return   # silently skip zero-contribution out-of-band
        self._data[i][k] += float(value)

    # ------------------------------------------------------------------
    # LDLT factorisation (Crout algorithm)
    # ------------------------------------------------------------------

    def factorize(self):
        """
        In-place LDLT (Crout) factorisation of the banded matrix.

        After calling this method the _data array stores L and D
        overwriting the original A.  The diagonal holds 1/D[i] for
        efficiency in the solve step.

        Inputs  : –  (uses self._data)
        Outputs : None  (modifies _data in-place)
        Raises  : ValueError if a zero or negative pivot is encountered.
        Assumptions : Matrix must be symmetric positive-definite.
        """
        n = self.n
        m = self.bandwidth

        # LDLT (Crout) factorisation of symmetric banded matrix.
        # _data[i][0] = D[i]  (diagonal)
        # _data[i][k] = L[i+k, i]  (upper band storage = lower factor by symmetry)

        D  = [0.0] * n   # diagonal factors
        L  = [[0.0] * (min(m, n - 1 - i) + 1) for i in range(n)]

        # Copy original data into L (L[i][0] is diagonal, L[i][k] is off-diag)
        for i in range(n):
            for k in range(len(self._data[i])):
                L[i][k] = self._data[i][k]

        for j in range(n):
            # D[j] = A[j,j] - sum_{k < j, |k-j|<=m}  L[j,k]^2 * D[k]
            d = L[j][0]
            istart = max(0, j - m)
            for k in range(istart, j):
                kj = j - k          # offset: L[k][kj] = L[k, k+kj] = A_orig[k, j] → L[j,k]
                if kj < len(L[k]):
                    Lkj = L[k][kj]
                    d  -= Lkj * Lkj * D[k]
            if abs(d) < 1e-14:
                raise ValueError(
                    f"BandedMatrix.factorize: zero pivot at index {j}. "
                    "Matrix may be singular or not positive-definite."
                )
            D[j] = d

            # L[i, j] = (A[i,j] - sum_{k< j} L[i,k]*L[j,k]*D[k]) / D[j]
            # For banded: i ranges from j+1 to min(j+m, n-1)
            for ii in range(j + 1, min(j + m + 1, n)):
                ij = ii - j         # offset: L[j][ij] = original A[j, j+ij] = A[j, ii]
                val = L[j][ij] if ij < len(L[j]) else 0.0
                istart_k = max(0, max(j, ii) - m)
                for k in range(istart_k, j):
                    kj  = j  - k    # L[k, j]
                    kii = ii - k    # L[k, ii]
                    Lkj  = L[k][kj]  if kj  < len(L[k]) else 0.0
                    Lkii = L[k][kii] if kii < len(L[k]) else 0.0
                    val -= Lkj * Lkii * D[k]
                val /= D[j]
                # Store into L[j][ij]  (upper band of j-th row → lower L of ii-th)
                if ij < len(L[j]):
                    L[j][ij] = val

        # Write back into _data: diagonal → D, off-diag → L
        self._D = D
        self._L = L
        self._factored = True

    # ------------------------------------------------------------------
    # Forward and back substitution
    # ------------------------------------------------------------------

    def solve(self, b: list) -> list:
        """
        Solve  A * x = b  after LDLT factorisation.

        If factorize() has not been called yet, it is called automatically.

        Inputs
        ------
        b : list[float] – right-hand-side, length n

        Outputs
        -------
        x : list[float] – solution vector, length n
        """
        if not self._factored:
            self.factorize()

        n  = self.n
        m  = self.bandwidth
        D  = self._D
        L  = self._L

        # Forward substitution:  L * y = b
        y = list(b)
        for i in range(n):
            istart = max(0, i - m)
            for k in range(istart, i):
                ki = i - k          # L[k][ki]  = L_{ik}  (lower triangle)
                if ki < len(L[k]):
                    y[i] -= L[k][ki] * y[k]

        # Diagonal scaling:  D * z = y
        z = [y[i] / D[i] for i in range(n)]

        # Backward substitution:  L^T * x = z
        x = list(z)
        for i in range(n - 1, -1, -1):
            m_i = min(m, n - 1 - i)
            for k in range(1, m_i + 1):
                if k < len(L[i]):
                    x[i] -= L[i][k] * x[i + k]

        return x

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def to_dense(self) -> list:
        """Return full n×n matrix as list-of-lists (for verification)."""
        n = self.n
        A = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for k in range(len(self._data[i])):
                j = i + k
                A[i][j] = self._data[i][k]
                if k > 0:
                    A[j][i] = self._data[i][k]   # symmetry
        return A

    def __repr__(self) -> str:
        return (f"BandedMatrix(n={self.n}, bandwidth={self.bandwidth}, "
                f"factored={self._factored})")
