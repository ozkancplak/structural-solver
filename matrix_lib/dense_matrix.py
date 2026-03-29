"""
dense_matrix.py
===============
Purpose : Full (dense) matrix stored as a list-of-lists.
          Provides basic linear-algebra operations and a
          built-in Gauss-elimination solver with partial pivoting.

Units   : dimensionless  (caller is responsible for consistent units)
Inputs  : (see individual method docstrings)
Outputs : (see individual method docstrings)

Notes
-----
* numpy is NOT used anywhere in this module.
* Symmetry storage: if symmetric=True only the upper triangle
  (including the diagonal) is stored; reads/writes to the lower
  triangle are transparently redirected.
"""

import math


class DenseMatrix:
    """
    A 2-D dense matrix stored as a Python list-of-lists.

    Parameters
    ----------
    rows : int   – number of rows
    cols : int   – number of columns
    symmetric : bool, optional (default False)
        If True, only the upper triangle is stored (rows × rows).
        The matrix is then implicitly symmetric: A[i,j] == A[j,i].
        *cols* is ignored when symmetric=True; the matrix is square.

    Attributes
    ----------
    rows : int
    cols : int
    symmetric : bool
    _data : list[list[float]]
    """

    # ------------------------------------------------------------------
    # Construction & helpers
    # ------------------------------------------------------------------

    def __init__(self, rows: int, cols: int, symmetric: bool = False):
        """
        Inputs  : rows (int), cols (int), symmetric (bool)
        Outputs : DenseMatrix instance initialised with zeros
        """
        self.rows = rows
        self.cols = rows if symmetric else cols
        self.symmetric = symmetric

        if symmetric:
            # Store only upper triangle: row i has (rows-i) entries
            self._data = [[0.0] * (rows - i) for i in range(rows)]
        else:
            self._data = [[0.0] * cols for _ in range(rows)]

    # --- index helpers -------------------------------------------------

    def _sym_index(self, i: int, j: int):
        """Return (row, col) in the upper-triangle storage."""
        if i <= j:
            return i, j - i
        else:
            return j, i - j          # swap to upper triangle

    # ------------------------------------------------------------------
    # Element access
    # ------------------------------------------------------------------

    def __getitem__(self, key):
        """
        A[i, j]  – zero-indexed element access.
        Inputs  : key = (i, j) tuple
        Outputs : float
        """
        i, j = key
        if self.symmetric:
            r, c = self._sym_index(i, j)
            return self._data[r][c]
        return self._data[i][j]

    def __setitem__(self, key, value: float):
        """
        A[i, j] = value  – zero-indexed element assignment.
        Inputs  : key = (i, j) tuple, value (float)
        Outputs : None  (in-place)
        """
        i, j = key
        if self.symmetric:
            r, c = self._sym_index(i, j)
            self._data[r][c] = float(value)
        else:
            self._data[i][j] = float(value)

    # ------------------------------------------------------------------
    # Basic arithmetic
    # ------------------------------------------------------------------

    def add(self, other: "DenseMatrix") -> "DenseMatrix":
        """
        Matrix addition: C = self + other.
        Inputs  : other (DenseMatrix) – must have same shape
        Outputs : DenseMatrix
        """
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("add: matrix dimensions do not match.")
        result = DenseMatrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result[i, j] = self[i, j] + other[i, j]
        return result

    def scale(self, alpha: float) -> "DenseMatrix":
        """
        Scalar multiplication: C = alpha * self.
        Inputs  : alpha (float)
        Outputs : DenseMatrix
        """
        result = DenseMatrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result[i, j] = alpha * self[i, j]
        return result

    def multiply(self, other: "DenseMatrix") -> "DenseMatrix":
        """
        Matrix multiplication: C = self @ other.
        Inputs  : other (DenseMatrix) – self.cols must equal other.rows
        Outputs : DenseMatrix
        """
        if self.cols != other.rows:
            raise ValueError("multiply: inner dimensions do not match.")
        result = DenseMatrix(self.rows, other.cols)
        for i in range(self.rows):
            for j in range(other.cols):
                s = 0.0
                for k in range(self.cols):
                    s += self[i, k] * other[k, j]
                result[i, j] = s
        return result

    def transpose(self) -> "DenseMatrix":
        """
        Matrix transpose: C = self^T.
        Inputs  : –
        Outputs : DenseMatrix (rows and cols swapped)
        """
        result = DenseMatrix(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                result[j, i] = self[i, j]
        return result

    def mat_vec(self, v: list) -> list:
        """
        Matrix-vector product: y = self * v.
        Inputs  : v (list[float]) – length self.cols
        Outputs : list[float] – length self.rows
        """
        if len(v) != self.cols:
            raise ValueError("mat_vec: vector length mismatch.")
        result = [0.0] * self.rows
        for i in range(self.rows):
            s = 0.0
            for j in range(self.cols):
                s += self[i, j] * v[j]
            result[i] = s
        return result

    # ------------------------------------------------------------------
    # Linear system solver (Gauss elimination with partial pivoting)
    # ------------------------------------------------------------------

    def solve(self, b: list) -> list:
        """
        Solve  A * x = b  via Gauss elimination with partial pivoting.

        Inputs
        ------
        b : list[float] – right-hand-side vector, length self.rows

        Outputs
        -------
        x : list[float] – solution vector

        Assumptions
        -----------
        * Matrix must be square (rows == cols).
        * numpy is NOT used.
        """
        n = self.rows
        if self.cols != n:
            raise ValueError("solve: matrix must be square.")
        if len(b) != n:
            raise ValueError("solve: b length must equal matrix rows.")

        # Build a mutable copy (augmented matrix [A | b])
        A = [[self[i, j] for j in range(n)] for i in range(n)]
        x = list(b)

        # Forward elimination with partial pivoting
        for col in range(n):
            # Find pivot
            max_val = abs(A[col][col])
            max_row = col
            for row in range(col + 1, n):
                if abs(A[row][col]) > max_val:
                    max_val = abs(A[row][col])
                    max_row = row

            if max_val < 1e-14:
                raise ValueError("solve: matrix is singular or nearly singular.")

            # Swap rows
            A[col], A[max_row] = A[max_row], A[col]
            x[col], x[max_row] = x[max_row], x[col]

            # Eliminate below
            for row in range(col + 1, n):
                factor = A[row][col] / A[col][col]
                x[row] -= factor * x[col]
                for c in range(col, n):
                    A[row][c] -= factor * A[col][c]

        # Back substitution
        solution = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = x[i]
            for j in range(i + 1, n):
                s -= A[i][j] * solution[j]
            solution[i] = s / A[i][i]

        return solution

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def copy(self) -> "DenseMatrix":
        """Return a deep copy."""
        result = DenseMatrix(self.rows, self.cols, self.symmetric)
        for i in range(self.rows):
            for j in range(self.cols):
                result[i, j] = self[i, j]
        return result

    def to_list(self) -> list:
        """Return full matrix as a list-of-lists (expanding symmetry)."""
        return [[self[i, j] for j in range(self.cols)] for i in range(self.rows)]

    def frobenius_norm(self) -> float:
        """Frobenius norm sqrt(sum(A_ij^2))."""
        s = 0.0
        for i in range(self.rows):
            for j in range(self.cols):
                s += self[i, j] ** 2
        return math.sqrt(s)

    def __repr__(self) -> str:
        lines = []
        for i in range(self.rows):
            row_str = "  ".join(f"{self[i,j]:12.5g}" for j in range(self.cols))
            lines.append(f"  [{row_str}]")
        sym_tag = " (symmetric)" if self.symmetric else ""
        return f"DenseMatrix({self.rows}×{self.cols}{sym_tag}):\n" + "\n".join(lines)
