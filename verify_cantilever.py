# -*- coding: utf-8 -*-
"""
verify_cantilever.py
====================
Purpose  : Verify the frame_analysis solver against the analytical solution
           for a cantilever beam with a tip point load.

Problem
-------
  L = 4 m,  EI = 200e9 × 1e-4 = 20,000,000 N·m²,  P = 10,000 N (downward)
  Analytical:
    v_tip   = -P*L^3 / (3*E*I) = -10000 * 64 / (3 * 2e7) = -1.0667e-2 m
    theta_tip = P*L^2 / (2*E*I) = 10000 * 16 / (2 * 2e7) = 4.0e-3 rad (counter-clockwise → positive)
    Reaction Ry at fixed end = +10000 N (upward)
    Reaction Mz at fixed end = +40000 N·m (counter-clockwise)

Outputs : PASS / FAIL per check, with tolerance 1e-4 (relative)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from read_input     import read_input
from frame_analysis import solve_frame


def check(name: str, computed: float, expected: float, tol: float = 1e-4):
    if abs(expected) < 1e-12:
        rel_err = abs(computed - expected)
    else:
        rel_err = abs((computed - expected) / expected)
    status = "PASS" if rel_err < tol else "FAIL"
    print(f"  [{status}]  {name}: computed={computed:.6e}  expected={expected:.6e}  rel_err={rel_err:.3e}")
    return status == "PASS"


def main():
    print("=" * 60)
    print("  Verification: Cantilever Beam (Analytical Solution)")
    print("=" * 60)

    data   = read_input(os.path.join(os.path.dirname(__file__), 'test_cantilever.txt'))
    result = solve_frame(data)

    E_arr = result['E']
    D     = result['D']

    # Node 2 displacements (0-indexed: node 1)
    u_eq = E_arr[1][0]; v_eq = E_arr[1][1]; r_eq = E_arr[1][2]

    v_tip     = D[v_eq - 1] if v_eq else 0.0
    theta_tip = D[r_eq - 1] if r_eq else 0.0

    P = 10000.0; L = 4.0; EI = 200e9 * 1e-4
    v_an     = -P * L**3 / (3 * EI)          # -1.0667e-2  (downward)
    theta_an = -P * L**2 / (2 * EI)          # -4.0e-3    (clockwise = negative)

    # Member forces at fixed end (element 0, local DOFs 0-2):
    # N_i = 0, V_i = +P (local upward balance), M_i = +P*L
    mf     = result['member_forces'][0]
    V_i    = mf[1]    # local shear at start (fixed end)
    M_i    = mf[2]    # local moment at start (fixed end)

    all_pass = True
    all_pass &= check("v_tip    [m]  ", v_tip,     v_an)
    all_pass &= check("theta_tip[rad]", theta_tip, theta_an)
    all_pass &= check("V_fixed  [N]  ", V_i,        P)
    all_pass &= check("M_fixed  [N.m]", M_i,        P * L)

    print("-" * 60)
    print("  Overall:", "ALL PASS" if all_pass else "SOME CHECKS FAILED")
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
