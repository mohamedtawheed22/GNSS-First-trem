import numpy as np

def design_matrix_poly(t: np.ndarray, p: int):
    """
    Description:
    Constructs the design matrix A for a polynomial of degree p on scaled time ∈ [-1,1].

    Input:
    t           [n x 1]  Time vector
    p           [1 x 1]  Polynomial degree

    Output:
    A           [n x (p+1)]  Design matrix
    scale       [1 x 1]      Scaling factor = (t_max - t_min) / 2
    t_sc        [n x 1]      Scaled time vector in [-1, 1]


    Author : Mohamed Amin
    Date   : June 2026
    """
    t_min, t_max = t.min(), t.max()
    scale = (t_max - t_min) / 2
    t_sc  = -1.0 + 2.0 * (t - t_min) / (t_max - t_min)
    A = np.column_stack([t_sc**k for k in range(p + 1)])
    return A, scale, t_sc


def design_matrix_harmonic(t: np.ndarray, p: int,
                             omega0: float, q: list):
    """
    Description:  Combined polynomial + harmonic design matrix.
    Constructs the design matrix A for a combined model consisting of:
           - a polynomial of degree p
           - harmonic functions at frequencies q * omega0

    Input:
    t       [n x 1]  Time vector in decimal years
    p       [1 x 1]  Polynomial degree
    omega0  [1 x 1]  Base angular frequency [rad/yr]
    q       [1 x m]  Vector of harmonic multiples

    Output:
    A       Design matrix
    scale       scaling factor
    t_scaled    scaled time vector
    
    Author : Mohamed Amin
    Date   : June 2026
    """
    A_poly, scale, t_sc = design_matrix_poly(t, p)
    cols = [A_poly]
    for j in q:
        cols.append(np.cos(j * omega0 * t).reshape(-1, 1))
        cols.append(np.sin(j * omega0 * t).reshape(-1, 1))
    A = np.hstack(cols)
    return A, scale, t_sc


def solve_lsa(A: np.ndarray, l: np.ndarray, sigma: np.ndarray):
    """
    Weighted least-squares adjustment.

    Input:
    A     : design matrix              [n x u]
    l     : observation vector         [n x 1]
    sigma : per-epoch sigma vector     [n x 1]  

    Output:
    xest  : estimated parameters       [u x 1]
    l_est : estimated observation      [n x 1]
    v     : residuals  (l_est - l)     [n x 1]
    Cxx   : covariance of parameters   [u x u]
    r_k   : redundancy components      [n x 1]

    Author : Mohamed Amin
    Date   : June 2026
    """
    p  = 1.0 / sigma**2            # per-epoch weights [n]
    PA = A * p[:, None]            # same as P @ A, without building the n x n matrix P

    N     = A.T @ PA               # normal equations
    n_vec = A.T @ (p * l)          

    Nxx  = np.linalg.inv(N)
    xest = Nxx @ n_vec       # estimated parameters 
 
    l_est = A @ xest         # estimated observation
    v     = l_est - l        # residuals
    Cxx   = Nxx              # covariance of parameters

    # Redundancy components: how much each observation is controlled by the others
    # 0 means the observation is necessary
    # 1 means the observation is fully controlled (an outlier there is easy to find)
    ANxx     = A @ Nxx                       # [n x u]
    diag_ANA = np.sum(ANxx * A, axis=1)      # [n]  diag(A Nxx A^T)
    r_k      = np.maximum(1.0 - p * diag_ANA, 1e-10)       

    return xest, l_est, v, Cxx, r_k
