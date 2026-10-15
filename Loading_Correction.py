import numpy as np
from scipy.interpolate import RegularGridInterpolator


def detrend(t: np.ndarray, y: np.ndarray):
    """
    Description:
    Remove a straight line (a + b*t) from a signal.

    Input:
    t : time vector                     [n]
    y : signal                          [n]

    Output:
    y_detrended : y - (a + b*t)         [n]

    Author : Mohamed Amin
    Date   : August 2026
    """
    coeffs = np.polyfit(t, y, 1)
    return y - np.polyval(coeffs, t)


def common_epochs(mjd_a: np.ndarray, mjd_b: np.ndarray):
    """
    Description:
    Find the epochs that exist in both time series (matched by integer MJD).

    Input:
    mjd_a : MJD of the first series     [n]
    mjd_b : MJD of the second series    [m]

    Output:
    idx_a : index of the common epochs in series a    [k]
    idx_b : index of the common epochs in series b    [k]

    Author : Mohamed Amin
    Date   : August 2026
    """
    _, idx_a, idx_b = np.intersect1d(np.round(mjd_a).astype(int),
                                     np.round(mjd_b).astype(int),
                                     return_indices=True)
    return idx_a, idx_b


def subtract_loading(data: dict, load: dict):
    """
    Description:
    Subtract NTOL and NTHL (HYDL) from the GNSS positions.

    Input:
    data : dict from load_tenv3     (m)
    load : dict from load_loading   (m)

    Output:
    result : dict with
             idx              index of the used GNSS epochs     [k]
             tp, t            time of the common epochs         [k]
             east, north, up  GNSS - NTOL - NTHL                [k]
             ntal             dict east/north/up of NTAL        [k]

    Author : Mohamed Amin
    Date   : August 2026
    """
    idx_g, idx_l = common_epochs(data['t'], load['t'])
    tp = data['tp'][idx_g]

    result = {'idx': idx_g, 'tp': tp, 't': data['t'][idx_g], 'ntal': {}}

    for comp, c in [('east', 'e'), ('north', 'n'), ('up', 'u')]:
        ntal = detrend(tp, load[f'ntal_{c}'][idx_l])
        ntol = detrend(tp, load[f'ntol_{c}'][idx_l])
        nthl = detrend(tp, load[f'hydl_{c}'][idx_l])

        result[comp]  = data[comp][idx_g] - ntol - nthl
        result['ntal'][comp] = ntal

    return result


def gia_at_stations(gia_file: str, lat: np.ndarray, lon: np.ndarray):
    """
    Description:
    Interpolate the GIA vertical land motion grid (Caron et al. 2018)
    at the station positions (bilinear interpolation).
    

    Input:
    gia_file : path of the GIA grid file          [str]
    lat, lon : station positions [deg]            [n]

    Output:
    vlm : GIA vertical rate at every station [mm/yr]   [n]

    Author : Mohamed Amin and AI for part of interpolation
    Date   : Augest 2026
    """
    grid = np.loadtxt(gia_file, comments='%', usecols=(0, 1, 2))
    g_lat = 90.0 - grid[:, 0]
    g_lon = np.where(grid[:, 1] > 180, grid[:, 1] - 360, grid[:, 1])
    g_vlm = grid[:, 2]

    # put the values into a regular (lat x lon) grid
    lats = np.unique(g_lat)
    lons = np.unique(g_lon)
    values = np.full((len(lats), len(lons)), np.nan)
    values[np.searchsorted(lats, g_lat), np.searchsorted(lons, g_lon)] = g_vlm

    interp = RegularGridInterpolator((lats, lons), values,
                                     bounds_error=False, fill_value=None)
    return interp(np.column_stack([lat, lon]))


def remove_gia(t: np.ndarray, up: np.ndarray, gia_rate: float):
    """
    Description:
    Remove the GIA vertical rate from the Up component (rate starts at the first epoch).

    Input:
    t        : decimal-year time vector                [n]
    up       : Up component (same unit as the rate * yr, e.g. mm)   [n]
    gia_rate : GIA vertical rate (e.g. mm/yr)          [1]

    Output:
    up_corr  : up - gia_rate * (t - t0)                [n]

    Author : Mohamed Amin
    Date   : August 2026
    """
    return up - gia_rate * (t - t[0])
