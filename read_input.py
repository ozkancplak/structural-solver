"""
read_input.py
=============
Purpose  : Parse a structured text file and return a problem data dict
           for frame_analysis.solve_frame().
Inputs   : filename (str) – path to input file
Outputs  : dict with keys NumNode, NumElem, XY, M, C, S, L, W

Input file format
-----------------
[NODES]         NumNode
  node_id  x  y

[MATERIALS]     NumMat
  mat_id  A  I  E

[CONNECTIVITY]  NumElem
  elem_id  start  end  mat_id  elem_type

[SUPPORTS]      NumSupport
  node_id  rx  ry  rz      (1=restrained)

[LOADS]         NumLoad
  node_id  Fx  Fy  Mz

[DIST_LOADS]    NumDistLoad     (optional)
  elem_id  w1  w2
"""


def read_input(filename: str) -> dict:
    """
    Parse the structured text file into a data dictionary.

    Inputs  : filename (str)
    Outputs : dict
    Assumptions : File encodes one section per keyword; blank lines / '#' comments ignored.
    """
    with open(filename, 'r') as fh:
        lines = [ln.strip() for ln in fh if ln.strip() and not ln.strip().startswith('#')]

    data = {}
    i = 0

    def next_tokens():
        """Return the next non-empty line split into tokens."""
        nonlocal i
        while i < len(lines):
            toks = lines[i].split()
            i += 1
            if toks:
                return toks
        return []

    while i < len(lines):
        line = lines[i]; i += 1
        kw = line.upper().split()[0] if line else ''

        if kw == '[NODES]':
            n = int(lines[i]); i += 1
            data['NumNode'] = n
            xy = []
            for _ in range(n):
                toks = next_tokens()
                xy.append([float(toks[1]), float(toks[2])])
            data['XY'] = xy

        elif kw == '[MATERIALS]':
            n = int(lines[i]); i += 1
            mats = []
            for _ in range(n):
                toks = next_tokens()
                mats.append([float(toks[1]), float(toks[2]), float(toks[3])])
            data['M'] = mats

        elif kw == '[CONNECTIVITY]':
            n = int(lines[i]); i += 1
            data['NumElem'] = n
            conn = []
            for _ in range(n):
                toks = next_tokens()
                conn.append([int(toks[1]), int(toks[2]), int(toks[3]), int(toks[4])])
            data['C'] = conn

        elif kw == '[SUPPORTS]':
            n = int(lines[i]); i += 1
            supp = []
            for _ in range(n):
                toks = next_tokens()
                supp.append([int(toks[0]), int(toks[1]), int(toks[2]), int(toks[3])])
            data['S'] = supp

        elif kw == '[LOADS]':
            n = int(lines[i]); i += 1
            loads = []
            for _ in range(n):
                toks = next_tokens()
                loads.append([int(toks[0]), float(toks[1]), float(toks[2]), float(toks[3])])
            data['L'] = loads

        elif kw == '[DIST_LOADS]':
            n = int(lines[i]); i += 1
            dl = []
            for _ in range(n):
                toks = next_tokens()
                dl.append([int(toks[0]), float(toks[1]), float(toks[2])])
            data['W'] = dl

    # Defaults for optional sections
    data.setdefault('W', [])
    data.setdefault('L', [])
    data.setdefault('S', [])
    return data
