import numpy as np
from scipy import stats


def global_test(v: np.ndarray, sigma: np.ndarray, redundancy: int, alpha: float):
    """
    Description:
    Chi-square global model test to check if the overall adjustment is accepted.

    Input:
    v           : residuals vector                 [n x 1]
    sigma       : sigma vector           [n x 1]
    redundancy  : degrees of freedom  (n - u)      [1 x 1]
    alpha       : significance level               [1 x 1]

    Output:
    rejected    : True if global test is rejected  [bool]
    T           : test statistic                   [1 x 1]
    k_crit      : chi-square critical value        [1 x 1]

    Author : Mohamed Amin
    Date   : June 2026
    """
    p      = 1.0 / sigma**2                              # weights
    T      = float((p * v**2).sum())                     # test statistic
    k_crit = stats.chi2.ppf(1.0 - alpha, df=redundancy)  # critical value
    return (T > k_crit), T, k_crit


def parameter_test(xest: np.ndarray, Cxx: np.ndarray, alpha: float):
    """
    Description:
    Individual parameter significance test.
    H0 : x_i = 0  (parameter is not significant)
    H1 : x_i ≠ 0  (parameter is significant)

    Input:
    xest    : estimated parameters vector          [u x 1]
    Cxx     : covariance matrix of parameters      [u x u]
    alpha   : significance level                   [1 x 1]

    Output:
    rejected    : True = parameter is significant  [u x 1]
    t_stat      : test statistic per parameter     [u x 1]
    k_crit      : normal critical value            [1 x 1]

    Author : Mohamed Amin
    Date   : June 2026
    """
    sigma_x  = np.sqrt(np.diag(Cxx))
    t_stat   = xest / sigma_x
    k_crit   = stats.norm.ppf(1.0 - alpha / 2.0)
    rejected = np.abs(t_stat) > k_crit
    return rejected, t_stat, k_crit


def outlier_test(v: np.ndarray, r_k: np.ndarray, sigma: np.ndarray,
                 u: int, global_rejected: bool, alpha: float):
    """
    Description:
    Outlier test for every observation with Bonferroni correction (alpha / n).
    Baarda (1968) : global test ACCEPTED -> sigma0 known     -> normal distribution
    Pope   (1976) : global test REJECTED -> sigma0 estimated -> tau distribution

    Input:
    v               : residuals vector                      [n x 1]
    r_k             : redundancy components                 [n x 1]
    sigma           : per-epoch sigma vector                [n x 1]
    u               : number of unknown parameters          [1 x 1]
    global_rejected : result of the global model test       [bool]
    alpha           : significance level                    [1 x 1]

    Output:
    is_outlier      : True = outlier                        [n x 1]
    tau             : test statistic per observation        [n x 1]
    k_crit          : critical value                        [1 x 1]

    Author : Mohamed Amin
    Date   : August 2026
    """
    n       = len(v)
    f       = n - u                 # degrees of freedom
    alpha_B = alpha / n             # Bonferroni correction
    v_std   = v / sigma

    if not global_rejected:
        # Baarda: standard normal
        tau    = np.abs(v_std) / np.sqrt(r_k)
        k_crit = stats.norm.ppf(1.0 - alpha_B / 2.0)
    else:
        # Pope: tau distribution
        sigma0_hat = np.sqrt(np.sum(v_std**2) / f)
        tau        = np.abs(v_std) / (sigma0_hat * np.sqrt(r_k))
        t_val      = stats.t.ppf(1.0 - alpha_B, df=f - 1)
        k_crit     = t_val * np.sqrt(f) / np.sqrt(f - 1 + t_val**2)

    is_outlier = tau > k_crit
    return is_outlier, tau, k_crit


def rms(x: np.ndarray):
    """
    Description:
    Root mean square of a vector.
 
    Input:
    x   : values        [n]
 
    Output:
    rms : RMS value     [1]
 
    Author : Mohamed Amin
    Date   : September 2026
    """
    return np.sqrt(np.mean(x**2))
 
 
def rms_reduction(before: np.ndarray, after: np.ndarray):
    """
    Description:
    RMS reduction in percent after a correction
    
    Input:
    before : signal before the correction    [n]
    after  : signal after the correction     [n]
 
    Output:
    reduction in %                           [1]
 
    Author : Mohamed Amin
    Date   : September 2026
    """
    return (1.0 - rms(after) / rms(before)) * 100.0