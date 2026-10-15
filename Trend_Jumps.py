import numpy as np
from DesignMatrix_LSA   import solve_lsa
from DesignMatrix_Jumps import clean_jumps, design_matrix_jumps, is_harmonic


def subtract_trend_jumps(data: dict, jumps: list, p: int = 1,
                         omega0: float = 2 * np.pi, q: list = None):
    """
    Description:
    Fit the jump-aware model (trend + harmonics + jumps) to E, N, U and
    subtract ONLY the trend and the jumps. T
    
    Input:
    data   : dict from Read_Wright_GNSS.load_tenv3
    jumps  : list of dicts with 'epoch' and 'type'
    p      : polynomial degree (1 = linear trend)       [int]
    omega0 : base angular frequency [rad/yr]            [float]
    q      : harmonic multiples                         [list]

    Output:
    resid   : dict east/north/up, observation - (trend + jumps)   [n]
    model   : dict east/north/up, the subtracted trend + jumps    [n]
    jumps_c : jump list really used in the model (after clean_jumps)

    Author : Mohamed Amin
    Date   : August 2026
    """
    if q is None:
        q = [1, 2]

    t        = data['tp']
    jumps_c  = clean_jumps(t, jumps, p, q)
    A, names = design_matrix_jumps(t, jumps_c, p, omega0, q)
    if A.shape[0] <= A.shape[1]:
        raise ValueError(f'not enough epochs: n={A.shape[0]}, u={A.shape[1]}')

    # columns that will be subtracted (everything except cos / sin)
    trend_cols = np.array([not is_harmonic(name) for name in names])

    resid = {}
    model = {}
    for comp, sig in [('east', 'sig_em'), ('north', 'sig_nm'), ('up', 'sig_um')]:
        xest, l_est, v, Cxx, r_k = solve_lsa(A, data[comp], data[sig])
        model[comp] = A[:, trend_cols] @ xest[trend_cols]
        resid[comp] = data[comp] - model[comp]

    return resid, model, jumps_c
