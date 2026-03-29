"""
verify_banded.py
================
Purpose  : Verify BandedMatrix LDLT solver against DenseMatrix Gauss
           elimination for a known 2D beam stiffness system.
Inputs   : – (hardcoded test cases)
Outputs  : PASS / FAIL per entry; printed to stdout
"""

import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))

from matrix_lib import DenseMatrix, BandedMatrix


def check(name: str, got: float, ref: float, tol: float = 1e-8) -> bool:
    if abs(ref) < 1e-12:
        err = abs(got - ref)
    else:
        err = abs((got - ref) / ref)
    status = "PASS" if err < tol else "FAIL"
    print(f"  [{status}]  {name}: banded={got:.8e}  dense={ref:.8e}  rel_err={err:.2e}")
    return status == "PASS"


def main():
    print("=" * 65)
    print("  Verification: BandedMatrix LDLT  vs  DenseMatrix Gauss")
    print("=" * 65)

    # Test 1: 4x4 symmetric banded SPD matrix (bandwidth=1, tridiagonal)
    # K = tridiagonal [2, -1; -1, 2; ...] scaled by 1000
    n, m = 4, 1
    vals = [
        (0,0, 2000.0), (0,1,-1000.0),
        (1,1, 2000.0), (1,2,-1000.0),
        (2,2, 2000.0), (2,3,-1000.0),
        (3,3, 2000.0),
    ]
    b = [1.0, 0.0, 0.0, 1.0]

    K_band  = BandedMatrix(n, m)
    K_dense = DenseMatrix(n, n)
    for i, j, v in vals:
        K_band[i, j]  = v
        K_dense[i, j] = v
        if i != j:
            K_dense[j, i] = v

    x_band  = K_band.solve(b[:])
    x_dense = K_dense.solve(b[:])

    print("\nTest 1: 4×4 tridiagonal (bandwidth=1)")
    all_pass = True
    for k in range(n):
        all_pass &= check(f"x[{k}]", x_band[k], x_dense[k])

    # Test 2: 6x6 SPD banded (bandwidth=2) – cantilever-like stiffness
    n2, m2 = 6, 2
    import random; random.seed(42)
    # Build a random diagonally dominant SPD banded matrix
    K2_band  = BandedMatrix(n2, m2)
    K2_dense = DenseMatrix(n2, n2)
    for i in range(n2):
        for k in range(m2 + 1):
            j = i + k
            if j >= n2:
                break
            v = random.uniform(0.5, 2.0) * (1 if k > 0 else 10.0)
            K2_band[i, j]  = v
            K2_dense[i, j] = v
            if k > 0:
                K2_dense[j, i] = v

    b2 = [float(i + 1) for i in range(n2)]
    x2_band  = K2_band.solve(b2[:])
    x2_dense = K2_dense.solve(b2[:])

    print("\nTest 2: 6×6 random SPD banded (bandwidth=2)")
    for k in range(n2):
        all_pass &= check(f"x[{k}]", x2_band[k], x2_dense[k])

    # Test 3: to_dense() round-trip
    print("\nTest 3: BandedMatrix.to_dense() round-trip")
    dense_from_band = K_band.to_dense()
    mismatch = False
    for i in range(n):
        for j in range(n):
            ref_v = K_dense[i, j]
            got_v = dense_from_band[i][j]
            if abs(got_v - ref_v) > 1e-12:
                print(f"  [FAIL] to_dense[{i},{j}]: {got_v} vs {ref_v}")
                mismatch = True
                all_pass = False
    if not mismatch:
        print("  [PASS]  to_dense() matches DenseMatrix reference exactly")

    print("-" * 65)
    print("  Overall:", "ALL PASS" if all_pass else "SOME CHECKS FAILED")
    print("=" * 65)
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
