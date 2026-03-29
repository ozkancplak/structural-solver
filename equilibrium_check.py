"""
equilibrium_check.py
====================
Purpose : Checks global structural equilibrium (Sum F_x = 0, Sum F_y = 0)
          using pure-Python lists (replaces week6 numpy version).

Units   : dimensionless
"""
import math

def calculate_reactions_and_check(data: dict, member_forces: list, element_matrices: list):
    """
    Computes nodal residuals to check overall static equilibrium.
    Returns (Reactions_dict, sum_Fx, sum_Fy, is_stable)
    """
    num_node = data['NumNode']
    Loads    = data['L']
    DistLoads = data.get('W', [])
    XY       = data['XY']

    # F_applied initialized to 0.0
    F_applied = [[0.0, 0.0, 0.0] for _ in range(num_node)]
    for row in Loads:
        node_idx = int(row[0]) - 1
        F_applied[node_idx][0] += row[1]
        F_applied[node_idx][1] += row[2]
        F_applied[node_idx][2] += row[3]

    # F_internal_sum = sum of all member end forces acting ON nodes
    F_internal_sum = [[0.0, 0.0, 0.0] for _ in range(num_node)]
    for i in range(data['NumElem']):
        f_local = member_forces[i]
        R = element_matrices[i]['R']
        start_node = int(element_matrices[i]['start_node']) - 1
        end_node   = int(element_matrices[i]['end_node']) - 1

        # f_global = R^T * f_local
        f_global = [sum(R[p][k] * f_local[p] for p in range(6)) for k in range(6)]

        F_internal_sum[start_node][0] += f_global[0]
        F_internal_sum[start_node][1] += f_global[1]
        F_internal_sum[start_node][2] += f_global[2]

        F_internal_sum[end_node][0] += f_global[3]
        F_internal_sum[end_node][1] += f_global[4]
        F_internal_sum[end_node][2] += f_global[5]

    # Reaction = F_internal - F_applied
    Reactions = [[0.0, 0.0, 0.0] for _ in range(num_node)]
    for nd in range(num_node):
        for dof in range(3):
            val = F_internal_sum[nd][dof] - F_applied[nd][dof]
            # Zero out tiny numbers
            Reactions[nd][dof] = 0.0 if abs(val) < 1e-9 else val

    # Global Equilibrium Check
    Total_Applied_X = sum(F_applied[nd][0] for nd in range(num_node))
    Total_Applied_Y = sum(F_applied[nd][1] for nd in range(num_node))

    # Add Distributed loads projections
    for row in DistLoads:
        mem_idx = int(row[0]) - 1
        w1 = row[1]
        w2 = row[2] if len(row) > 2 else w1
        
        c = element_matrices[mem_idx]['R'][0][0]
        s = element_matrices[mem_idx]['R'][0][1]
        
        sn = int(element_matrices[mem_idx]['start_node']) - 1
        en = int(element_matrices[mem_idx]['end_node']) - 1
        dx = XY[en][0] - XY[sn][0]
        dy = XY[en][1] - XY[sn][1]
        L  = math.sqrt(dx**2 + dy**2)

        total_w = (w1 + w2) / 2.0 * L
        
        # In local coords, w is in y direction. 
        # Global projection: Wx = Wlocal_y * (-sin), Wy = Wlocal_y * cos
        global_Wx = total_w * (-s)
        global_Wy = total_w * c
        
        Total_Applied_X += global_Wx
        Total_Applied_Y += global_Wy

    Total_Reaction_X = sum(Reactions[nd][0] for nd in range(num_node))
    Total_Reaction_Y = sum(Reactions[nd][1] for nd in range(num_node))

    sum_Fx = Total_Applied_X + Total_Reaction_X
    sum_Fy = Total_Applied_Y + Total_Reaction_Y

    is_stable = (abs(sum_Fx) < 1e-5) and (abs(sum_Fy) < 1e-5)

    return Reactions, sum_Fx, sum_Fy, is_stable
