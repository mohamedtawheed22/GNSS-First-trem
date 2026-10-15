import numpy as np
from DesignMatrix_LSA   import solve_lsa
from DesignMatrix_Jumps import clean_jumps, design_matrix_jumps
from Statistical_Test   import global_test, outlier_test


def iterative_outlier_removal(t: np.ndarray, l: np.ndarray, sigma: np.ndarray,
                              jumps: list = None, p: int = 1,
                              omega0: float = 2 * np.pi, q: list = None,
                              alpha: float = 0.05, max_iter: int = 20):
    """
    Description:
    Iteratively detect and remove outliers for one GNSS component.
    Every iteration:
      1. clean the jump list for the remaining epochs
      2. build the jump-aware design matrix and solve the weighted LSA
      3. global chi-square test -> choose Baarda or Pope
      4. flag the outliers and remove them
    Stops when no outlier is found or max_iter is reached.

    Input:
    t        : decimal-year time vector                 [n]
    l        : observation vector                       [n]
    sigma    : per-epoch sigma vector                   [n]
    jumps    : list of dicts with 'epoch' and 'type'
    p        : polynomial degree                        [int]
    omega0   : base angular frequency [rad/yr]          [float]
    q        : harmonic multiples                       [list]
    alpha    : significance level                       [float]
    max_iter : maximum number of iterations             [int]

    Output:
    outlier  : True = epoch is an outlier               [n]

    Author : Mohamed Amin
    Date   : September 2026
    """
    if jumps is None:
        jumps = []
    if q is None:
        q = [1, 2]

    outlier = np.zeros(len(l), dtype=bool)

    for it in range(max_iter):
        keep = ~outlier
        t_w, l_w, s_w = t[keep], l[keep], sigma[keep]

        # 1-2. model for the remaining epochs
        jumps_w = clean_jumps(t_w, jumps, p, q)
        A, _    = design_matrix_jumps(t_w, jumps_w, p, omega0, q)
        n, u    = A.shape
        if n <= u:
            break
        xest, l_est, v, Cxx, r_k = solve_lsa(A, l_w, s_w)

        # 3-4. tests
        rejected, T, k_chi = global_test(v, s_w, n - u, alpha)
        is_out, tau, k_out = outlier_test(v, r_k, s_w, u, rejected, alpha)
        if is_out.sum() == 0:
            break

        idx = np.where(keep)[0]          # map back to the original epochs
        outlier[idx[is_out]] = True

    return outlier


def outlier_removal_station(data: dict, jumps: list, p: int = 1,
                            omega0: float = 2 * np.pi, q: list = None,
                            alpha: float = 0.05, max_iter: int = 20):
    """
    Description:
    Run iterative_outlier_removal on East, North and Up.
    An epoch is removed if it is an outlier in at least one component.

    Input:
    data   : dict from Read_Wright_GNSS.load_tenv3
    jumps  : list of dicts with 'epoch' and 'type'
    p, omega0, q, alpha, max_iter : model / test settings

    Output:
    outlier : True = epoch removed                      [n]
    counts  : dict with number of outliers per component

    Author : Mohamed Amin
    Date   : September 2026
    """
    out_e = iterative_outlier_removal(data['tp'], data['east'],  data['sig_em'],
                                      jumps, p, omega0, q, alpha, max_iter)
    out_n = iterative_outlier_removal(data['tp'], data['north'], data['sig_nm'],
                                      jumps, p, omega0, q, alpha, max_iter)
    out_u = iterative_outlier_removal(data['tp'], data['up'],    data['sig_um'],
                                      jumps, p, omega0, q, alpha, max_iter)

    outlier = out_e | out_n | out_u
    counts  = {'east' : int(out_e.sum()),
               'north': int(out_n.sum()),
               'up'   : int(out_u.sum()),
               'total': int(outlier.sum())}
    return outlier, counts
