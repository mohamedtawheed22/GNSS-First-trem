"""
Jump-aware design matrix for GNSS LSA.

Jump types:
  type 1 : model change -> a new segment starts (own constant, trend, harmonics)
  type 0 : offset only  -> one step column (0 before the jump, 1 after it)

Author : Mohamed Amin and the logic for treatment of jumps at edgas and too close jumps enhanced by AI
Date   : August 2026
"""
import numpy as np
from DesignMatrix_LSA import design_matrix_harmonic


def clean_jumps(t: np.ndarray, jumps: list, p: int = 1, q: list = None, min_edge: float = 0.02):
    """
    Description:
    Prepare the jump list before building the design matrix:
      1. remove jumps too close to the start or the end of the data
      2. a type-1 jump is changed to type-0 if the data before it
         or after it is too short 

    Input:
    t        : decimal-year time vector                 [n]
    jumps    : list of dicts with 'epoch' and 'type'
    p        : polynomial degree                        [int]
    q        : harmonic multiples                       [list]
    min_edge : minimum distance to the data edges [yr]  [float]

    Output:
    new_jumps : cleaned list of jumps, sorted by epoch

    Author : Mohamed Amin enhanced by AI
    Date   : August 2026
    """
    if q is None:
        q = [1, 2]
    u_seg = 1 + p + 2 * len(q)

    # 1. remove jumps at the edges
    new_jumps = [dict(j) for j in jumps if t.min() + min_edge < j['epoch'] < t.max() - min_edge]
    new_jumps.sort(key=lambda j: j['epoch'])

    # 2. check the segment length of every type-1 jump
    last = t.min()
    for j in new_jumps:
        if j['type'] == 1:
            n_before = np.sum((t >= last) & (t < j['epoch']))
            n_after  = np.sum(t >= j['epoch'])
            if n_before > u_seg and n_after > u_seg:
                last = j['epoch']       # good jump -> new segment starts here
            else:
                j['type'] = 0           # too short -> offset only

    return new_jumps


def design_matrix_jumps(t: np.ndarray, jumps: list, p: int = 1,
                        omega0: float = 2 * np.pi, q: list = None):
    """
    Description:
    Jump-aware design matrix.
      type-1 jump : new segment, every segment has its own polynomial + harmonics
      type-0 jump : one step column (1 after the jump, inside its segment)
    

    Input:
    t       : decimal-year time vector                  [n]
    jumps   : cleaned list of jumps
    p       : polynomial degree                         [int]
    omega0  : base angular frequency [rad/yr]           [float]
    q       : harmonic multiples ([] = no harmonics)    [list]

    Output:
    A       : design matrix                             [n x u]
    names   : name of every column                      [u]

    Author : Mohamed Amin
    Date   : July 2026
    """
    if q is None:
        q = [1, 2]

    # segment number of every epoch
    seg = np.zeros(len(t), dtype=int)
    for j in jumps:
        if j['type'] == 1:
            seg[t >= j['epoch']] += 1

    cols  = []
    names = []

    # polynomial + harmonics for every segment
    for s in range(seg.max() + 1):
        A_s, _, _ = design_matrix_harmonic(t[seg == s], p, omega0, q)
        col_names = [f't^{k}' for k in range(p + 1)]
        for j in q:
            col_names += [f'cos{j}', f'sin{j}']

        for k in range(A_s.shape[1]):
            col = np.zeros(len(t))
            col[seg == s] = A_s[:, k]
            cols.append(col)
            names.append(f'seg{s}_{col_names[k]}')

    # step column for every type-0 jump
    for j in jumps:
        if j['type'] == 0:
            s    = seg[t >= j['epoch']][0]                  # segment of the jump
            step = ((t >= j['epoch']) & (seg == s)) * 1.0
            if 0 < step.sum() < np.sum(seg == s):           # data before AND after the jump
                cols.append(step)
                names.append(f"jump_{j['epoch']:.4f}")

    A = np.column_stack(cols)
    return A, names


def is_harmonic(name: str):
    """
    Description:
    True if the column name is a cos / sin column. this part is important for the subtract_trend_jumps() function.

    Author : Mohamed Amin
    Date   : August 2026
    """
    return ('cos' in name) or ('sin' in name)