"""
frame_analysis.py
=================
Purpose : 2-D frame/truss analysis using the matrix stiffness method.
          Replaces numpy with the custom matrix_lib library.

          Supports element types:
            1 – Standard plane frame member
            2 – Frame member with hinge at start node (i-end)
            3 – Frame member with hinge at end node   (j-end)
            4 – Truss member (axial stiffness only)

Inputs  :  data (dict) – parsed problem data, keys:
              NumNode, NumElem, XY, M, C, S, L, W (optional)
Outputs :  dict with keys:
              E, NumEq, K_skyline, F, D, member_forces, element_matrices

Units   :  Caller is responsible for consistent force / length units.

Notes
-----
* Global stiffness K is assembled into a SkylineMatrix.
* Element 6×6 local/global matrices use plain Python lists (not DenseMatrix)
  for speed in small dense operations.
* numpy is NOT used here (import is only allowed in the verification script).
"""

import math
from matrix_lib import SkylineMatrix, LinearSolver


# ======================================================================
# Internal helpers
# ======================================================================

def _zeros(n: int) -> list:
    """Return a 1-D zero list of length n."""
    return [0.0] * n


def _zeros2(r: int, c: int) -> list:
    """Return an r×c zero list-of-lists."""
    return [[0.0] * c for _ in range(r)]


def _mat_mat_6(A: list, B: list) -> list:
    """6×6 matrix multiplication  C = A @ B  using pure Python."""
    C = _zeros2(6, 6)
    for i in range(6):
        for j in range(6):
            s = 0.0
            for k in range(6):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C


def _mat_transpose_mat_6(R: list, K: list) -> list:
    """Compute R^T @ K @ R for 6×6 matrices."""
    # First compute tmp = K @ R
    tmp = _mat_mat_6(K, R)
    # Then compute R^T @ tmp
    result = _zeros2(6, 6)
    for i in range(6):
        for j in range(6):
            s = 0.0
            for k in range(6):
                s += R[k][i] * tmp[k][j]   # R^T[i,k] = R[k,i]
            result[i][j] = s
    return result


def _mat_vec_6(A: list, v: list) -> list:
    """6×6 matrix times 6-vector."""
    return [sum(A[i][k] * v[k] for k in range(6)) for i in range(6)]


# ======================================================================
# Equation numbering
# ======================================================================

def build_equation_numbers(num_node: int, supports: list):
    """
    Assign global equation numbers to free DOFs.

    Inputs
    ------
    num_node : int       – number of nodes
    supports : list      – rows of [node_id, rx, ry, rz]  (1-indexed node)
                           rx/ry/rz = 1 means restrained

    Outputs
    -------
    E     : list[list[int]]  – shape (num_node, 3), equation number (1-based) or 0
    NumEq : int              – total number of free DOFs
    """
    E_bc = [[0, 0, 0] for _ in range(num_node)]

    for row in supports:
        node_idx = int(row[0]) - 1
        E_bc[node_idx][0] = int(row[1])
        E_bc[node_idx][1] = int(row[2])
        E_bc[node_idx][2] = int(row[3])

    E = [[0, 0, 0] for _ in range(num_node)]
    eq_num = 1
    for r in range(num_node):
        for c in range(3):
            if E_bc[r][c] == 0:
                E[r][c] = eq_num
                eq_num += 1

    return E, eq_num - 1


# ======================================================================
# Singularity check
# ======================================================================

def check_truss_singularities(num_node: int, connectivity: list, supports: list):
    """
    Detect truss nodes whose rotation DOF is undefined.

    Inputs
    ------
    num_node     : int
    connectivity : list – rows of [start, end, mat_id, elem_type]
    supports     : list – rows of [node_id, rx, ry, rz]

    Outputs
    -------
    None – raises ValueError with a descriptive message on failure.
    """
    node_types = {i: set() for i in range(1, num_node + 1)}
    for row in connectivity:
        nd_i = int(row[0])
        nd_j = int(row[1])
        etype = int(row[3]) if len(row) > 3 else 1
        node_types[nd_i].add(etype)
        node_types[nd_j].add(etype)

    s_dict = {int(row[0]): row for row in supports}
    errors = []
    for node, types in node_types.items():
        if len(types) == 1 and 4 in types:
            if node not in s_dict or int(s_dict[node][3]) == 0:
                errors.append(node)

    if errors:
        msg = (
            "\n" + "=" * 70 + "\n"
            "ERROR: TRUSS NODE ROTATION SINGULARITY!\n"
            "=" * 70 + "\n"
            f"Nodes connected ONLY to truss elements: {errors}\n"
            "Add rotation restraint (rz=1) for these nodes in [SUPPORTS].\n"
            "=" * 70 + "\n"
        )
        raise ValueError(msg)


# ======================================================================
# Element stiffness matrices
# ======================================================================

def create_element_matrices(start_node: int, end_node: int,
                            XY: list, M: list,
                            mat_id: int, elem_type: int):
    """
    Build local, rotation, and global 6×6 stiffness matrices.

    Inputs
    ------
    start_node : int   – 1-indexed start node
    end_node   : int   – 1-indexed end node
    XY         : list  – node coordinates [[x0,y0], [x1,y1], …]
    M          : list  – material properties [[A, I, E], …]  (0-indexed)
    mat_id     : int   – 1-indexed material ID
    elem_type  : int   – 1=frame, 2=hinge-start, 3=hinge-end, 4=truss

    Outputs
    -------
    k_local  : list[list[float]] – 6×6 local stiffness
    R        : list[list[float]] – 6×6 rotation matrix
    k_global : list[list[float]] – 6×6 global stiffness  (R^T k_local R)
    """
    x1, y1 = XY[int(start_node) - 1]
    x2, y2 = XY[int(end_node)   - 1]

    A_sec, I_sec, E_mod = M[int(mat_id) - 1]

    dx = x2 - x1
    dy = y2 - y1
    L  = math.sqrt(dx * dx + dy * dy)
    if L < 1e-12:
        raise ValueError(f"Element from node {start_node} to {end_node} has zero length.")
    c  = dx / L
    s  = dy / L

    # ---- Local stiffness ----
    k = _zeros2(6, 6)

    # Axial terms (all types)
    EA_L = E_mod * A_sec / L
    k[0][0] =  EA_L;  k[0][3] = -EA_L
    k[3][0] = -EA_L;  k[3][3] =  EA_L

    EI = E_mod * I_sec
    if elem_type == 1:
        k[1][1] =  12*EI/L**3;  k[1][4] = -12*EI/L**3
        k[4][1] = -12*EI/L**3;  k[4][4] =  12*EI/L**3
        k[1][2] =   6*EI/L**2;  k[1][5] =   6*EI/L**2
        k[2][1] =   6*EI/L**2;  k[2][4] =  -6*EI/L**2
        k[4][2] =  -6*EI/L**2;  k[4][5] =  -6*EI/L**2
        k[5][1] =   6*EI/L**2;  k[5][4] =  -6*EI/L**2
        k[2][2] =   4*EI/L;     k[2][5] =   2*EI/L
        k[5][2] =   2*EI/L;     k[5][5] =   4*EI/L

    elif elem_type == 2:   # hinge at start
        k[1][1] =  3*EI/L**3;  k[1][4] = -3*EI/L**3
        k[4][1] = -3*EI/L**3;  k[4][4] =  3*EI/L**3
        k[1][5] =  3*EI/L**2;  k[5][1] =  3*EI/L**2
        k[4][5] = -3*EI/L**2;  k[5][4] = -3*EI/L**2
        k[5][5] =  3*EI/L

    elif elem_type == 3:   # hinge at end
        k[1][1] =  3*EI/L**3;  k[1][4] = -3*EI/L**3
        k[4][1] = -3*EI/L**3;  k[4][4] =  3*EI/L**3
        k[1][2] =  3*EI/L**2;  k[2][1] =  3*EI/L**2
        k[2][4] = -3*EI/L**2;  k[4][2] = -3*EI/L**2
        k[2][2] =  3*EI/L

    elif elem_type == 4:   # truss (axial only – already set)
        pass

    # ---- Rotation matrix 6×6 ----
    R = _zeros2(6, 6)
    R[0][0] =  c;  R[0][1] = s
    R[1][0] = -s;  R[1][1] = c
    R[2][2] =  1.0
    R[3][3] =  c;  R[3][4] = s
    R[4][3] = -s;  R[4][4] = c
    R[5][5] =  1.0

    # k_global = R^T @ k_local @ R
    k_global = _mat_transpose_mat_6(R, k)

    return k, R, k_global


# ======================================================================
# Main solver
# ======================================================================

def solve_frame(data: dict) -> dict:
    """
    Solve a 2-D frame/truss using the matrix stiffness method.

    Inputs
    ------
    data : dict with keys:
        NumNode (int), NumElem (int),
        XY      (list[[x,y]]),
        M       (list[[A, I, E]]),
        C       (list[[start, end, mat_id, elem_type]]),
        S       (list[[node, rx, ry, rz]]),
        L       (list[[node, Fx, Fy, Mz]]),
        W       (list[[elem_id, w1, w2]]) – optional distributed loads

    Outputs
    -------
    dict with keys:
        E              – equation number array (list[list[int]])
        NumEq          – number of free DOFs (int)
        K_skyline      – assembled SkylineMatrix (unfactored)
        F              – global load vector (list[float])
        D              – displacement vector (list[float])
        member_forces  – list of 6-element local force lists per element
        element_matrices – list of dicts {k_local, R, start_node, end_node}
    """
    num_node = data['NumNode']
    num_elem = data['NumElem']
    XY       = data['XY']
    M        = data['M']
    C        = data['C']
    S        = data['S']
    Loads    = data['L']
    DistLoad = data.get('W', [])

    # 0. Pre-flight checks
    check_truss_singularities(num_node, C, S)

    # 1. Equation numbering
    E, num_eq = build_equation_numbers(num_node, S)

    # 2. Distributed load equivalent nodal loads
    ENL = {}                          # mem_id → {local: list, global: list}
    for row in DistLoad:
        mem_id = int(row[0])
        w1, w2 = row[1], row[2]

        mem_idx    = mem_id - 1
        sn = int(C[mem_idx][0]);  en = int(C[mem_idx][1])
        x1, y1 = XY[sn - 1];     x2, y2 = XY[en - 1]
        dx = x2 - x1;             dy = y2 - y1
        L  = math.sqrt(dx*dx + dy*dy)
        c  = dx / L;              s  = dy / L

        R = _zeros2(6, 6)
        R[0][0]=c; R[0][1]=s; R[1][0]=-s; R[1][1]=c; R[2][2]=1
        R[3][3]=c; R[3][4]=s; R[4][3]=-s; R[4][4]=c; R[5][5]=1

        # Fixed-end forces (trapezoidal load, local y-direction)
        V1 = (7*w1 + 3*w2) * L / 20.0
        V2 = (3*w1 + 7*w2) * L / 20.0
        M1 = (3*w1 + 2*w2) * L*L / 60.0
        M2 = -(2*w1 + 3*w2) * L*L / 60.0

        f_loc = [0.0, V1, M1, 0.0, V2, M2]
        f_glo = _mat_vec_6(
            [[R[j][i] for j in range(6)] for i in range(6)],   # R^T
            f_loc
        )
        ENL[mem_id] = {'local': f_loc, 'global': f_glo}

    # 3. Global stiffness matrix (SkylineMatrix)
    K = SkylineMatrix(num_eq)

    # Determine skyline profile from element connectivity
    element_matrices = []
    eq_nums_all = []

    for i in range(num_elem):
        sn  = int(C[i][0]);  en  = int(C[i][1])
        mat = int(C[i][2]);  ety = int(C[i][3])

        k_loc, R_mat, k_glo = create_element_matrices(sn, en, XY, M, mat, ety)
        element_matrices.append({
            'k_local': k_loc, 'R': R_mat,
            'start_node': sn, 'end_node': en
        })

        u1 = E[sn-1][0]; v1 = E[sn-1][1]; r1 = E[sn-1][2]
        u2 = E[en-1][0]; v2 = E[en-1][1]; r2 = E[en-1][2]
        eqs = [u1, v1, r1, u2, v2, r2]
        eq_nums_all.append(eqs)

        # Expand skyline profile
        active = [eq - 1 for eq in eqs if eq != 0]  # 0-indexed
        for p in active:
            for q in active:
                if p >= q:
                    K.expand_profile(q, p)          # row q, col p (col-major)

    # Assemble K (upper triangle only to avoid double-counting in symmetric storage)
    for i, eqs in enumerate(eq_nums_all):
        k_glo = element_matrices[i]
        # Recalculate k_global from stored data
        k_g   = _mat_transpose_mat_6(k_glo['R'], k_glo['k_local'])
        for p in range(6):
            P = eqs[p]
            if P == 0:
                continue
            for q in range(6):
                Q = eqs[q]
                if Q == 0:
                    continue
                # Only add upper triangle (P <= Q); SkylineMatrix is symmetric
                if P <= Q:
                    K.add_to(P - 1, Q - 1, k_g[p][q])

    # 4. Global load vector
    F = _zeros(num_eq)
    for row in Loads:
        ni = int(row[0]) - 1
        Fx, Fy, Mz = row[1], row[2], row[3]
        u = E[ni][0]; v = E[ni][1]; r = E[ni][2]
        if u: F[u-1] += Fx
        if v: F[v-1] += Fy
        if r: F[r-1] += Mz

    # Add equivalent nodal loads from distributed loads
    for mem_id, enl in ENL.items():
        idx  = mem_id - 1
        sn   = int(C[idx][0]); en = int(C[idx][1])
        eqs  = eq_nums_all[idx]
        f_g  = enl['global']
        for p in range(6):
            P = eqs[p]
            if P:
                F[P-1] += f_g[p]

    # 5. Solve  K * D = F
    K_copy = _clone_skyline(K)   # keep original for reaction computation
    try:
        D = K_copy.solve(F)
    except ValueError as exc:
        msg = (
            "\n" + "=" * 70 + "\n"
            "ERROR: UNSTABLE STRUCTURE (SINGULAR STIFFNESS MATRIX)\n"
            "=" * 70 + "\n"
            + str(exc) + "\n"
            "Check [SUPPORTS]: ensure sufficient restraints exist.\n"
            "=" * 70 + "\n"
        )
        raise ValueError(msg)

    # 6. Member end forces
    member_forces = []
    for i in range(num_elem):
        eqs   = eq_nums_all[i]
        elem  = element_matrices[i]

        d_glo = [D[eq - 1] if eq != 0 else 0.0 for eq in eqs]
        d_loc = _mat_vec_6(elem['R'], d_glo)
        f_loc = _mat_vec_6(elem['k_local'], d_loc)

        if (i + 1) in ENL:
            fe = ENL[i + 1]['local']
            f_loc = [f_loc[p] - fe[p] for p in range(6)]

        member_forces.append(f_loc)

    return {
        'E':               E,
        'NumEq':           num_eq,
        'K_skyline':       K,
        'F':               F,
        'D':               D,
        'member_forces':   member_forces,
        'element_matrices': element_matrices,
        'eq_nums_all':     eq_nums_all
    }


def _clone_skyline(K: SkylineMatrix) -> SkylineMatrix:
    """Deep-copy a SkylineMatrix (before factorisation)."""
    n    = K.n
    Kc   = SkylineMatrix(n)
    Kc._height = list(K._height)
    Kc._col    = [list(col) for col in K._col]
    return Kc
