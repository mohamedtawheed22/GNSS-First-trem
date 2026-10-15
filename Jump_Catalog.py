"""
Find the discontinuities (jumps) of a GNSS station from:
  - the NGL steps file  (equipment changes + earthquakes found by NGL)
  - a USGS earthquake catalog (optional, fills the gaps)

Author : Mohamed Amin
Date   : July 2026
"""
import json
import numpy as np
import pandas as pd
from datetime import datetime


def decimal_year(date):
    """
    Description:
    Convert a date (datetime or pandas Timestamp) to decimal year.

    Input:
    date : datetime / Timestamp

    Output:
    decimal year                              [float]

    Author : Mohamed Amin
    Date   : July 2026
    """
    date = pd.Timestamp(date)
    if date.tzinfo is not None:
        date = date.tz_localize(None)       # remove the time zone (USGS times are UTC)
    start = datetime(date.year, 1, 1)
    end   = datetime(date.year + 1, 1, 1)
    return date.year + (date - start).total_seconds() / (end - start).total_seconds()


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Description:
    Great-circle distance with the haversine formula.
    Works for numbers and numpy arrays.

    Input:
    lat1, lon1 : first point(s)  [deg]
    lat2, lon2 : second point(s) [deg]

    Output:
    distance                     [km]

    Author : Mohamed Amin
    Date   : July 2026 
    """
    R    = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def distance_threshold(mag: float):
    """
    Description:
    Maximum distance where an earthquake can still make a jump,
    NGL formula:  d = 10^(M/2 - 0.79)  [km]
    with limits: M >= 8.5 -> max 15000 km, M >= 7.5 -> max 5000 km,
                 M < 5.0  -> min 30 km

    Input:
    mag : earthquake magnitude                [float]

    Output:
    d   : distance threshold                  [km]

    Author : Mohamed Amin
    Date   : July 2026
    """
    d = 10 ** (mag / 2.0 - 0.79)
    if mag >= 8.5:
        d = min(d, 15000.0)
    elif mag >= 7.5:
        d = min(d, 5000.0)
    elif mag < 5.0:
        d = max(d, 30.0)
    return d


def earthquake_type(mag: float, dist: float):
    """
    Description:
    Jump type of an earthquake.
    Type 1 (model change): M >= 8.0
                           M >= 7.0 and distance < 500 km
                           M >= 6.0 and distance < 100 km
    Type 0 (offset only) : all other earthquakes

    Input:
    mag  : magnitude                          [float]
    dist : station - epicenter distance [km]  [float]

    Output:
    0 or 1

    Author : Mohamed Amin
    Date   : July 2026
    """
    if mag >= 8.0:
        return 1
    if mag >= 7.0 and dist < 500.0:
        return 1
    if mag >= 6.0 and dist < 100.0:
        return 1
    return 0


def equipment_type(label: str):
    """
    Description:
    Jump type of an equipment change from its NGL label.
    The type-0 words are checked first (e.g. 'receiver make' is only a mode change).
    Type 0 : antenna, radome, elevation, gull, sn changed, firmware ...
    Type 1 : receiver, monument, site change, volcanic, unknown ...
    Not known label -> type 1 (safe side)

    Input:
    label : NGL event text                    [str]

    Output:
    0 or 1

    Author : Mohamed Amin
    Date   : July 2026
    """
    label = label.lower()
    type0_words = ['receiver make', 'receiver mode', 'antenna', 'radome',
                   'elevation', 'gull', 'sn changed', 'setting corrected', 'firmware']
    type1_words = ['receiver', 'logfile', 'monument', 'site change', 'volcanic',
                   'eruption', 'equipment site', 'unknown', 'unexplained',
                   'unspecified', 'possible', 'cabled', 'detached', 'mount changed', 'pin']

    for word in type0_words:
        if word in label:
            return 0
    for word in type1_words:
        if word in label:
            return 1
    return 1


def load_ngl_steps(filepath: str):
    """
    Description:
    Read the NGL steps file. Two kinds of lines:
      equipment : STA  YYMMMDD  1  Event_Label
      earthquake: STA  YYMMMDD  2  threshold  distance  magnitude  usgs_id

    Input:
    filepath : path of the steps file         [str]

    Output:
    steps : DataFrame with station, epoch, is_eq, label, mag, dist

    Author : Mohamed Amin
    Date   : June 2026
    """
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                epoch = decimal_year(pd.to_datetime(parts[1], format='%y%b%d'))
            except ValueError:
                continue

            if len(parts) >= 7:          # earthquake line
                try:
                    rows.append([parts[0].upper(), epoch, True, parts[6],
                                 float(parts[5]), float(parts[4])])
                    continue
                except ValueError:
                    pass
            # equipment line
            rows.append([parts[0].upper(), epoch, False,
                         parts[3].replace('_', ' '), np.nan, np.nan])

    return pd.DataFrame(rows, columns=['station', 'epoch', 'is_eq', 'label', 'mag', 'dist'])


def load_usgs_catalog(csv_path: str, mag_min: float = 4.5):
    """
    Description:
    Read a USGS earthquake CSV and add the decimal year.

    Input:
    csv_path : path of the USGS csv            [str]
    mag_min  : smallest magnitude to keep      [float]

    Output: 
    eq : DataFrame with latitude, longitude, mag, epoch

    Author : Mohamed Amin
    Date   : July 2026
    """
    eq = pd.read_csv(csv_path)
    eq = eq[eq['mag'] >= mag_min].copy()
    eq['epoch'] = [decimal_year(d) for d in pd.to_datetime(eq['time'], utc=True)]
    return eq[['latitude', 'longitude', 'mag', 'epoch']].reset_index(drop=True)


def merge_close_jumps(jumps: list, min_sep: float = 0.03):
    """
    Description:
    Sort the jumps and merge jumps closer than min_sep (same event reported
    twice, e.g. by NGL and USGS). The merged jump keeps the higher type
    and the larger earthquake.

    Input:
    jumps   : list of jump dicts
    min_sep : merge window [yr]               [~11 days]

    Output:
    merged  : list of jump dicts

    Author : Mohamed Amin
    Date   : July 2026
    """
    jumps  = sorted(jumps, key=lambda j: j['epoch'])
    merged = []
    for j in jumps:
        if len(merged) > 0 and j['epoch'] - merged[-1]['epoch'] < min_sep:
            last = merged[-1]
            last['label'] = last['label'] + ' + ' + j['label']
            last['type']  = max(last['type'], j['type'])
            if last['source'] != j['source']:
                last['source'] = 'both'
            if j['mag'] is not None and (last['mag'] is None or j['mag'] > last['mag']):
                last['mag']  = j['mag']
                last['dist'] = j['dist']
        else:
            merged.append(dict(j))
    return merged


def find_jumps(station: str, lat: float, lon: float, t_start: float, t_end: float,
               steps: pd.DataFrame, eq: pd.DataFrame = None,
               mag_min: float = 5.0, min_sep: float = 0.03):
    """
    Description:
    Find all jumps of one station inside its observation time:
      a) NGL steps file: equipment changes and earthquakes
      b) USGS catalog  : earthquakes closer than distance_threshold(M)
    Close jumps are merged with merge_close_jumps().

    Input:
    station         : 4-char station name             [str]
    lat, lon        : station position [deg]          [float]
    t_start, t_end  : first / last epoch [yr]         [float]
    steps           : DataFrame from load_ngl_steps
    eq              : DataFrame from load_usgs_catalog (or None)
    mag_min         : smallest USGS magnitude used    [float]
    min_sep         : merge window and edge margin    [yr]

    Output:
    jumps : list of dicts with epoch, type, source, label, mag, dist

    Author : Mohamed Amin
    Date   : August 2026
    """
    jumps = []

    # a) NGL steps file
    sta_steps = steps[steps['station'] == station.upper()]
    for _, row in sta_steps.iterrows():
        if row['is_eq']:
            jumps.append({'epoch': row['epoch'], 'source': 'earthquake',
                          'label': f"M{row['mag']:.1f} @ {row['dist']:.0f}km ({row['label']})",
                          'type' : earthquake_type(row['mag'], row['dist']),
                          'mag'  : row['mag'], 'dist': row['dist']})
        else:
            jumps.append({'epoch': row['epoch'], 'source': 'equipment',
                          'label': row['label'],
                          'type' : equipment_type(row['label']),
                          'mag'  : None, 'dist': None})

    # b) USGS catalog
    if eq is not None and not np.isnan(lat):
        eq_s = eq[(eq['mag'] >= mag_min) & (eq['epoch'] > t_start) & (eq['epoch'] < t_end)]
        dist = haversine_distance(lat, lon, eq_s['latitude'].values, eq_s['longitude'].values)
        mags = eq_s['mag'].values
        for k in range(len(eq_s)):
            if dist[k] <= distance_threshold(mags[k]):
                jumps.append({'epoch': eq_s['epoch'].values[k], 'source': 'earthquake',
                              'label': f"M{mags[k]:.1f} @ {dist[k]:.0f}km (USGS)",
                              'type' : earthquake_type(mags[k], dist[k]),
                              'mag'  : float(mags[k]), 'dist': float(dist[k])})

    # keep only jumps inside the data, then merge
    jumps = [j for j in jumps if t_start + min_sep < j['epoch'] < t_end - min_sep]
    return merge_close_jumps(jumps, min_sep)


