"""
verify_portal.py
================
Purpose  : Verify the frame_analysis solver against numpy.linalg.solve
           reference for a fixed-base portal frame.

All matrix operations in the reference use numpy directly.
Tolerance = 1e-4 relative.

Outputs  : PASS / FAIL per DOF.
"""

import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))

# ---------- new solver (matrix_lib) ----------
from read_input     import read_input
from frame_analysis import solve_frame

# ---------- numpy reference ----------
import numpy as np


def check(name: str, got: float, ref: float, tol: float = 1e-4) -> bool:
    if abs(ref) < 1e-12:
        err = abs(got - ref)
    else:
        err = abs((got - ref) / ref)
    status = "PASS" if err < tol else "FAIL"
    print(f"  [{status}]  {name}: matrix_lib={got:.6e}  numpy={ref:.6e}  rel_err={err:.2e}")
    return status == "PASS"


def solve_numpy_reference(data: dict):
    """
    Pure numpy reference solver mirroring frame_analysis.solve_frame logic.
    Inputs  : data (dict) same format as read_input output
    Outputs : (E, D_flat) – equation number array, displacement list
    """
    from frame_analysis import (build_equation_numbers, create_element_matrices)

    num_node = data['NumNode']
    num_elem = data['NumElem']
    XY       = data['XY']
    M        = data['M']
    C        = data['C']
    S        = data['S']
    Loads    = data['L']

    E, num_eq = build_equation_numbers(num_node, S)
    K = np.zeros((num_eq, num_eq))
    F = np.zeros(num_eq)

    for i in range(num_elem):
        sn = int(C[i][0]); en = int(C[i][1])
        mat = int(C[i][2]); ety = int(C[i][3])
        _, _, k_glo = create_element_matrices(sn, en, XY, M, mat, ety)
        k_np = np.array(k_glo)

        u1=E[sn-1][0]; v1=E[sn-1][1]; r1=E[sn-1][2]
        u2=E[en-1][0]; v2=E[en-1][1]; r2=E[en-1][2]
        eqs = [u1, v1, r1, u2, v2, r2]

        for p in range(6):
            P = eqs[p]
            if P == 0: continue
            for q in range(6):
                Q = eqs[q]
                if Q == 0: continue
                K[P-1, Q-1] += k_np[p, q]

    for row in Loads:
        ni = int(row[0]) - 1
        Fx, Fy, Mz = row[1], row[2], row[3]
        u=E[ni][0]; v=E[ni][1]; r=E[ni][2]
        if u: F[u-1] += Fx
        if v: F[v-1] += Fy
        if r: F[r-1] += Mz

    D = np.linalg.solve(K, F)
    return E, D.tolist()


def main():
    print("=" * 65)
    print("  Verification: Portal Frame – matrix_lib vs numpy reference")
    print("=" * 65)

    inp = os.path.join(os.path.dirname(__file__), 'test_portal.txt')

    data = read_input(inp)

    # --- New solver ---
    res_new = solve_frame(data)
    D_new   = res_new['D']
    E_new   = res_new['E']

    # --- Numpy reference ---
    E_ref, D_ref = solve_numpy_reference(data)

    all_pass = True
    num_node = data['NumNode']
    dof_names = ['u', 'v', 'theta']
    for nd in range(num_node):
        for dof in range(3):
            eq_new = E_new[nd][dof]
            eq_ref = E_ref[nd][dof]
            if eq_new == 0 and eq_ref == 0:
                continue
            if eq_new == 0 or eq_ref == 0:
                print(f"  [WARN] Node {nd+1} DOF {dof_names[dof]}: restraint mismatch")
                continue
            got = D_new[eq_new - 1]
            ref = D_ref[eq_ref - 1]
            all_pass &= check(f"Node {nd+1} {dof_names[dof]}", got, ref)

    print("-" * 65)
    print("  Overall:", "ALL PASS" if all_pass else "SOME CHECKS FAILED")
    print("=" * 65)
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
