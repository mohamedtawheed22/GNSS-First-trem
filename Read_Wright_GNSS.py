import numpy as np


def load_tenv3(filepath: str):
    """
    Description:
    the code is based in Dr. Lucks matlab code. Read a NGL tenv3 file 
    Read a NGL tenv3 file (IGS14, IGS20 or load predictions).
    The columns are separated by spaces, so every line is split and the
    needed columns are taken by their index:
        2  decimal year         3  MJD
        7  e0 + 8  east         9  n0 + 10 north       11 u0 + 12 up
        14 sig_e   15 sig_n   16 sig_u
        20 latitude   21 longitude
    Input:
    filepath : path of the tenv3 file                        [str]

    Output:
    data : dict with
           tp, t                     decimal year, MJD       [n]
           east, north, up           positions (m)           [n]
           sig_em, sig_nm, sig_um    sigmas (m)              [n]
           lat, lon                  station position (deg)  [1]
           header                    first line of the file  [str]
           lines                     data lines split in words (for writing back)

    Author : Mohamed Amin
    Date   : May 2026
    """
    with open(filepath, 'r') as f:
        all_lines = f.readlines()

    header = all_lines[0]
    lines  = []
    for line in all_lines[1:]:
        parts = line.split()
        if len(parts) < 17:
            continue
        try:
            float(parts[2])          # header or broken line -> skip
        except ValueError:
            continue
        lines.append(parts)

    def column(idx):
        return np.array([float(parts[idx]) for parts in lines])

    data = {
        'tp'     : column(2),
        't'      : column(3),
        'east'   : column(7)  + column(8),
        'north'  : column(9)  + column(10),
        'up'     : column(11) + column(12),
        'sig_em' : column(14),
        'sig_nm' : column(15),
        'sig_um' : column(16),
        'header' : header,
        'lines'  : lines,
    }

    # station position from the first line (if the columns exist)
    data['lat'] = np.nan
    data['lon'] = np.nan
    if len(lines) > 0 and len(lines[0]) >= 22:
        data['lat'] = float(lines[0][20])
        data['lon'] = float(lines[0][21])
        if data['lon'] > 180:
            data['lon'] = data['lon'] - 360
    return data


def subset_data(data: dict, keep: np.ndarray):
    """
    Description:
    Keep only the epochs where keep is True 

    Input:
    data : dict from load_tenv3                              [dict]
    keep : boolean mask, True = keep the epoch               [n]

    Output:
    new_data : same dict with fewer epochs                   [dict]

    Author : Mohamed Amin
    Date   : September 2026
    """
    new_data = dict(data)
    for key in ['tp', 't', 'east', 'north', 'up', 'sig_em', 'sig_nm', 'sig_um']:
        new_data[key] = data[key][keep]
    new_data['lines'] = [parts for parts, k in zip(data['lines'], keep) if k]
    return new_data


def replace_enu(data: dict, east: np.ndarray, north: np.ndarray, up: np.ndarray):
    """
    Description:
    Put new E/N/U values (e.g. residuals) into the data lines.
    The reference columns e0, n0, u0 are set to 0, so reading the file
    again with load_tenv3 gives exactly the new values.

    Input:
    data              : dict from load_tenv3                 [dict]
    east, north, up   : new values (m)                       [n]

    Output:
    new_data : dict with the new values and new lines        [dict]

    Author : Mohamed Amin
    Date   : September 2026
    """
    new_lines = []
    for i, parts in enumerate(data['lines']):
        parts = list(parts)
        parts[7],  parts[8]  = '0', f'{east[i]:.6f}'
        parts[9],  parts[10] = '0', f'{north[i]:.6f}'
        parts[11], parts[12] = '0', f'{up[i]:.6f}'
        new_lines.append(parts)

    new_data = dict(data)
    new_data['east']  = east
    new_data['north'] = north
    new_data['up']    = up
    new_data['lines'] = new_lines
    return new_data


def save_tenv3(filepath: str, data: dict):
    """
    Description:
    Write a tenv3 file (header + data lines).

    Input:
    filepath : where to save the file                        [str]
    data     : dict from load_tenv3 / subset_data / replace_enu

    Output:
    file saved in filepath

    Author : Mohamed Amin
    Date   : May 2026
    """
    with open(filepath, 'w') as f:
        f.write(data['header'])
        for parts in data['lines']:
            f.write(' '.join(parts) + '\n')


def load_loading(filepath: str):
    """
    Description:
    Read the loading columns of a NGL load prediction file
    (tenv3_loadpredictions). The loading values are after the 23
    standard tenv3 columns:
        23-25  NTAL  (E, N, U)  non-tidal atmospheric loading
        26-28  NTOL  (E, N, U)  non-tidal ocean loading
        29-31  HYDL  (E, N, U)  hydrological loading
        32-34  GRACE (E, N, U)  GRACE mascon loading

    Input:
    filepath : path of the load prediction file              [str]

    Output:
    load : dict with tp, t and ntal_e ... grace_u  (m)       [n]

    Author : Mohamed Amin
    Date   : July 2026
    """
    names = ['ntal_e', 'ntal_n', 'ntal_u', 'ntol_e', 'ntol_n', 'ntol_u',
             'hydl_e', 'hydl_n', 'hydl_u', 'grace_e', 'grace_n', 'grace_u']

    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 35:
                continue
            try:
                rows.append([float(x) for x in [parts[2], parts[3]] + parts[23:35]])
            except ValueError:
                continue

    rows = np.array(rows)
    load = {'tp': rows[:, 0], 't': rows[:, 1]}
    for k, name in enumerate(names):
        load[name] = rows[:, k + 2]
    return load

