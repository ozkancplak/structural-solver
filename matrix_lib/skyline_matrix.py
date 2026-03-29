"""
skyline_matrix.py
=================
Purpose : Symmetric skyline (profile / variable-bandwidth) matrix.
          Stores only the active columns up to the first non-zero
          in each column, resulting in optimal storage for sparse
          symmetric positive-definite matrices arising in FEM.

Units   : dimensionless  (caller is responsible for consistent units)
Inputs  : (see individual method docstrings)
Outputs : (see individual method docstrings)

Storage layout
--------------
For column j the stored entries run from row  (j - height[j] + 1)
up to and including row j (the diagonal).

  _col[j]  is a list of length  height[j].
  _col[j][k]  = A[j - height[j] + 1 + k, j]   for k = 0 … height[j]-1
  _col[j][-1] = A[j, j]   (diagonal, last element of each column list)

The column height is determined automatically during assembly by
tracking the highest (smallest row index) non-zero in each column.

Notes
-----
* numpy is NOT used anywhere in this module.
* Column heights are allowed to grow dynamically during assembly.
"""

import math


class SkylineMatrix:
    """
    Symmetric skyline (profile) matrix with dynamic column height tracking.

    Parameters
    ----------
    n : int – matrix order (n × n)

    Attributes
    ----------
    n       : int
    _height : list[int]    – height of each skyline column
    _col    : list[list]   – stored entries per column (see layout above)
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, n: int):
        """
        Inputs  : n (int)
        Outputs : SkylineMatrix initialised with zeros; height[j]=1 (diagonal only)
        """
        self.n = n
        self._height  = [1] * n        # at least diagonal is stored
        self._col     = [[0.0] for _ in range(n)]   # each col has 1 entry = diagonal
        self._factored = False

    # ------------------------------------------------------------------
    # Profile expansion before assembly
    # ------------------------------------------------------------------

    def expand_profile(self, i: int, j: int):
        """
        Ensure the skyline profile covers entry (i, j).
        Expands the column height of column max(i,j) if necessary.

        Inputs  : i, j (int) – row and column indices (0-based)
        Outputs : None  (modifies _height and _col in-place)
        """
        if i > j:
            i, j = j, i              # work with upper triangle: col=j, row=i
        col   = j
        required_height = j - i + 1  # row i means height ≥ (j-i+1)
        if required_height > self._height[col]:
            extra = required_height - self._height[col]
            self._col[col]    = [0.0] * extra + self._col[col]
            self._height[col] = required_height

    # ------------------------------------------------------------------
    # Element access
    # ------------------------------------------------------------------

    def _col_row_to_index(self, col: int, row: int):
        """
        Convert (col, row) to index within _col[col].
        Returns None if (row, col) is out of skyline profile.
        """
        h = self._height[col]
        top_row = col - h + 1         # first stored row of this column
        if row < top_row:
            return None
        return row - top_row

    def __getitem__(self, key) -> float:
        """
        A[i, j]  – zero-indexed symmetric access.
        Inputs  : key = (i, j) tuple
        Outputs : float  (0.0 for entries outside skyline profile)
        """
        i, j = key
        if i > j:
            i, j = j, i              # symmetry
        idx = self._col_row_to_index(j, i)
        if idx is None:
            return 0.0
        return self._col[j][idx]

    def __setitem__(self, key, value: float):
        """
        A[i, j] = value.
        The profile is automatically expanded if needed.
        Inputs  : key = (i, j) tuple, value (float)
        Outputs : None  (in-place)
        """
        i, j = key
        if i > j:
            i, j = j, i
        self.expand_profile(i, j)
        idx = self._col_row_to_index(j, i)
        self._col[j][idx] = float(value)

    def add_to(self, i: int, j: int, value: float):
        """
        A[i, j] += value  (main assembly accumulation method).
        Inputs  : i, j (int);  value (float)
        Outputs : None  (in-place)
        """
        if i > j:
            i, j = j, i
        self.expand_profile(i, j)
        idx = self._col_row_to_index(j, i)
        self._col[j][idx] += float(value)

    # ------------------------------------------------------------------
    # LDLT factorisation (column-by-column Crout for skyline)
    # ------------------------------------------------------------------

    def factorize(self):
        """
        In-place LDLT (Crout) factorisation.

        After this call:
          * diagonal entries _col[j][-1] hold D[j]
          * off-diagonal entries hold the L factors

        Inputs  : –  (uses _col, _height)
        Outputs : None  (modifies _col in-place; sets _factored=True)
        Raises  : ValueError on zero or negative pivot.
        Assumptions : Matrix is symmetric positive-definite.
        """
        n = self.n

        for j in range(n):
            h    = self._height[j]
            top  = j - h + 1          # first row stored in column j

            # --- Compute column j of L and D[j] ---
            # For each row i from top to j-1, compute  L[i,j] then update diagonal
            d_j = self._col[j][-1]    # start with A[j,j]

            for k_loc in range(h - 1):
                i = top + k_loc           # actual row index

                # L[i, j] = (A[i,j] - sum_{p < i within skyline} L[i,p]*D[p]*L[j,p]) / D[i]
                # Here: p runs from max(top_i, top_j) to i-1

                h_i   = self._height[i]
                top_i = i - h_i + 1
                top_common = max(top_i, top)   # start of overlap between col i and col j

                Aij = self._col[j][k_loc]      # current A[i,j] (upper col j, row i)

                for p in range(top_common, i):
                    # L[p, i] stored in column i → index (p - top_i)  of _col[i]
                    idx_pi = p - top_i
                    if idx_pi < 0 or idx_pi >= h_i:
                        continue
                    Lpi = self._col[i][idx_pi]

                    # L[p, j] stored in column j → index (p - top) of _col[j]
                    idx_pj = p - top
                    if idx_pj < 0 or idx_pj >= h - 1:
                        continue
                    Lpj = self._col[j][idx_pj]

                    # D[p] = diagonal of column p = _col[p][-1]
                    Dp  = self._col[p][-1]

                    Aij -= Lpi * Dp * Lpj

                # D[i] is stored as _col[i][-1]
                Di = self._col[i][-1]
                if abs(Di) < 1e-14:
                    raise ValueError(
                        f"SkylineMatrix.factorize: zero pivot at index {i}."
                    )
                L_ij = Aij / Di
                self._col[j][k_loc] = L_ij

                # Update D[j]
                d_j -= L_ij * L_ij * Di

            if abs(d_j) < 1e-14:
                raise ValueError(
                    f"SkylineMatrix.factorize: zero pivot at column {j}. "
                    "Matrix may be singular or not positive-definite."
                )
            self._col[j][-1] = d_j   # store D[j] in diagonal position

        self._factored = True

    # ------------------------------------------------------------------
    # Forward / diagonal / backward substitution
    # ------------------------------------------------------------------

    def solve(self, b: list) -> list:
        """
        Solve  A * x = b  using LDLT factors.

        Inputs
        ------
        b : list[float] – right-hand-side, length n

        Outputs
        -------
        x : list[float] – solution vector, length n
        """
        if not self._factored:
            self.factorize()

        n = self.n
        y = list(b)

        # Forward substitution:  L * y = b
        for j in range(1, n):
            h   = self._height[j]
            top = j - h + 1
            for k_loc in range(h - 1):
                i = top + k_loc
                L_ij = self._col[j][k_loc]
                y[j] -= L_ij * y[i]

        # Diagonal scaling:  D * z = y
        z = [y[j] / self._col[j][-1] for j in range(n)]

        # Backward substitution:  L^T * x = z
        x = list(z)
        for j in range(n - 1, -1, -1):
            h   = self._height[j]
            top = j - h + 1
            for k_loc in range(h - 1):
                i = top + k_loc
                L_ij = self._col[j][k_loc]
                x[i] -= L_ij * x[j]

        return x

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def to_dense(self) -> list:
        """Return full n×n matrix as list-of-lists (for verification)."""
        n = self.n
        A = [[0.0] * n for _ in range(n)]
        for j in range(n):
            h   = self._height[j]
            top = j - h + 1
            for k_loc in range(h):
                i = top + k_loc
                val = self._col[j][k_loc]
                A[i][j] = val
                A[j][i] = val
        return A

    def storage_size(self) -> int:
        """Total number of stored floats."""
        return sum(self._height)

    def __repr__(self) -> str:
        return (f"SkylineMatrix(n={self.n}, "
                f"stored_entries={self.storage_size()}, "
                f"factored={self._factored})")
