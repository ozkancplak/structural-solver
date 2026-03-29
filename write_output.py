"""
write_output.py
===============
Purpose  : Format and print analysis results to console and/or file.
Inputs   : result dict from frame_analysis.solve_frame(), data dict
Outputs  : formatted text to stdout and optional file
Units    : same as input (caller's responsibility)
"""


def write_output(result: dict, data: dict, output_file: str = None):
    """
    Print and optionally save analysis results.

    Inputs
    ------
    result      : dict from solve_frame()
    data        : original input data dict
    output_file : str or None – if given, also write to this path

    Outputs
    -------
    None  (side-effect: prints to stdout / writes file)
    """
    lines = []

    def p(text: str = ''):
        lines.append(text)
        print(text)

    E            = result['E']
    D            = result['D']
    member_forces = result['member_forces']
    num_node     = data['NumNode']
    num_elem     = data['NumElem']
    XY           = data['XY']
    C            = data['C']
    S_list       = data['S']

    # ---- Header ----
    p("=" * 72)
    p("  CE 4011 – Frame Analysis Results  (matrix_lib: SkylineMatrix)")
    p("=" * 72)

    # ---- Nodal displacements ----
    p("\nNODAL DISPLACEMENTS")
    p("-" * 50)
    p(f"{'Node':>6}  {'u (horiz)':>15}  {'v (vert)':>15}  {'rot (rad)':>15}")
    p("-" * 50)
    for nd in range(num_node):
        u_eq = E[nd][0]; v_eq = E[nd][1]; r_eq = E[nd][2]
        u_val = D[u_eq - 1] if u_eq else 0.0
        v_val = D[v_eq - 1] if v_eq else 0.0
        r_val = D[r_eq - 1] if r_eq else 0.0
        p(f"{nd+1:>6}  {u_val:>15.6e}  {v_val:>15.6e}  {r_val:>15.6e}")

    # ---- Support reactions ----
    p("\nSUPPORT REACTIONS")
    p("-" * 50)
    p(f"{'Node':>6}  {'Rx':>15}  {'Ry':>15}  {'Mz':>15}")
    p("-" * 50)

    support_dict = {int(row[0]): row for row in S_list}
    K_sky  = result['K_skyline']
    eq_all = result['eq_nums_all']

    for nd, row in support_dict.items():
        rx, ry, rz = int(row[1]), int(row[2]), int(row[3])
        Rx = Ry = Mz = '-'

        # Reaction = K_full * D - applied forces  … simplified via K*u for restrained DOFs
        # For output purposes use a direct computation from global K (stored unfactored):
        # We compute reaction by summing row contributions of global K * D.
        # Access K via to_dense() – acceptable for small problems.
        if not hasattr(K_sky, '_K_dense_cache'):
            K_sky._K_dense_cache = K_sky.to_dense()
        K_dense = K_sky._K_dense_cache
        n_eq = result['NumEq']

        nd_idx = nd - 1
        E_nd = E[nd_idx]

        def _reaction(dof_local, is_restrained):
            if not is_restrained:
                return '-'
            eq = E_nd[dof_local]
            # This DOF is restrained → eq == 0
            # Reaction = row sum of raw K * D for the ORIGINAL row (before BCs)
            # We cannot directly access restrained row from SkylineMatrix.
            # Use the ENL/member-force-based approach instead.
            return 'computed'   # placeholder – computed below

        # Direct reaction via member forces: sum member end forces at restrained node
        # Transformation: F_global = R^T * F_local  (R^T[i,j] = R[j,i])
        Rx_val = 0.0; Ry_val = 0.0; Mz_val = 0.0
        for e_idx, eqs in enumerate(eq_all):
            sn = int(C[e_idx][0]); en = int(C[e_idx][1])
            mf = member_forces[e_idx]    # local forces at start(0-2) and end(3-5)
            R_mat = result['element_matrices'][e_idx]['R']
            # F_global = R^T @ F_local  (note: R[p][k] = R^T[k][p])
            f_g = [sum(R_mat[p][k] * mf[p] for p in range(6)) for k in range(6)]
            if sn == nd:
                Rx_val += f_g[0]; Ry_val += f_g[1]; Mz_val += f_g[2]
            if en == nd:
                Rx_val += f_g[3]; Ry_val += f_g[4]; Mz_val += f_g[5]

        rx_str = f"{Rx_val:>15.4f}" if rx else f"{'(free)':>15}"
        ry_str = f"{Ry_val:>15.4f}" if ry else f"{'(free)':>15}"
        mz_str = f"{Mz_val:>15.4f}" if rz else f"{'(free)':>15}"
        p(f"{nd:>6}  {rx_str}  {ry_str}  {mz_str}")

    # ---- Member end forces ----
    p("\nMEMBER END FORCES  (local coordinates)")
    p("-" * 72)
    p(f"{'Elem':>5}  {'N_i':>11}  {'V_i':>11}  {'M_i':>11}  {'N_j':>11}  {'V_j':>11}  {'M_j':>11}")
    p("-" * 72)
    for e_idx, mf in enumerate(member_forces):
        p(f"{e_idx+1:>5}  "
          f"{mf[0]:>11.4f}  {mf[1]:>11.4f}  {mf[2]:>11.4f}  "
          f"{mf[3]:>11.4f}  {mf[4]:>11.4f}  {mf[5]:>11.4f}")

    # ---- Global equilibrium check ----
    p("\nEQUILIBRIUM CHECK  (Sum of applied loads + reactions)")
    p("-" * 50)
    sum_Fx = 0.0; sum_Fy = 0.0
    for row in data.get('L', []):
        sum_Fx += row[1]; sum_Fy += row[2]
    for nd, row in support_dict.items():
        rx, ry = int(row[1]), int(row[2])
        R_x_val = 0.0; R_y_val = 0.0
        for e_idx, eqs in enumerate(eq_all):
            sn = int(C[e_idx][0]); en = int(C[e_idx][1])
            mf = member_forces[e_idx]
            R_mat = result['element_matrices'][e_idx]['R']
            f_g = [sum(R_mat[p][k] * mf[p] for p in range(6)) for k in range(6)]
            if sn == nd:
                R_x_val += f_g[0]; R_y_val += f_g[1]
            if en == nd:
                R_x_val += f_g[3]; R_y_val += f_g[4]
        if rx: sum_Fx += R_x_val
        if ry: sum_Fy += R_y_val

    ref_mag = max(abs(r) for row in data.get('L', [[0,0,1,0]]) for r in [row[1], row[2]]) or 1.0
    tol = 1e-6 * ref_mag
    ok = abs(sum_Fx) < tol and abs(sum_Fy) < tol
    p(f"  SumFx = {sum_Fx:+.4e}   SumFy = {sum_Fy:+.4e}   [{'OK' if ok else 'WARNING'}]")

    p("\n" + "=" * 72)
    p("  Analysis complete.")
    p("=" * 72)

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(lines) + '\n')
        print(f"\n  Results written to: {output_file}")
