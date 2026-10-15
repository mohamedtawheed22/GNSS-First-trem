import time
import numpy as np
import pandas as pd
import requests
from io import StringIO
from pathlib import Path
from Jump_Catalog import haversine_distance

HOLDINGS_URL = 'https://geodesy.unr.edu/NGLStationPages/DataHoldings.txt'


# the following code for downloading and filtering the NGL station list part of the work form Anne klos,
# Dommink and Amin and enhanced by AI for the part of filtering and writing the station lists.

def load_data_holdings(source: str = HOLDINGS_URL):
    """
    Description:
    Read the NGL station list DataHoldings.txt 

    Input:
    source : URL or local path of DataHoldings.txt         [str]

    Out
    stations : DataFrame with station, lat, lon, height, begin, end, num_sol

    Author : Mohamed Amin
    Date   : August 2026
    """
    if source.startswith('http'):
        text = requests.get(source, timeout=120).text
    else:
        text = Path(source).read_text()

    rows = []
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 11:
            continue
        try:
            lon = float(parts[2])
            if lon > 180:
                lon = lon - 360
            rows.append([parts[0].upper(), float(parts[1]), lon, float(parts[3]),
                         parts[7], parts[8], int(parts[10])])
        except ValueError:
            continue

    stations = pd.DataFrame(rows, columns=['station', 'lat', 'lon', 'height',
                                           'begin', 'end', 'num_sol'])
    return stations.drop_duplicates('station').reset_index(drop=True)


def stations_in_box(stations: pd.DataFrame, lat_min: float, lat_max: float,
                    lon_min: float, lon_max: float, min_solutions: int = 0):
    """
    Description:
    Keep the stations inside a lat/lon box with enough daily solutions.

    Input:
    stations         : DataFrame from load_data_holdings
    lat_min, lat_max : latitude limits [deg]
    lon_min, lon_max : longitude limits [deg]
    min_solutions    : minimum number of daily solutions (1825 ~ 5 years)

    Output:
    DataFrame with the selected stations

    Author : Mohamed Amin
    Date   : September 2026
    """
    keep = ((stations['lat'] >= lat_min) & (stations['lat'] <= lat_max) &
            (stations['lon'] >= lon_min) & (stations['lon'] <= lon_max) &
            (stations['num_sol'] >= min_solutions))
    return stations[keep].reset_index(drop=True)


def stations_in_shapefile(stations: pd.DataFrame, shapefile: str):
    """
    Description:
    Keep the stations inside the polygons of a shapefile
    
    Input:
    stations  : DataFrame from load_data_holdings
    shapefile : path of the .shp file                  [str]

    Output:
    DataFrame with the stations inside the shape

    Author : Mohamed Amin / Anne Klos / dommink
    Date   : September 2026
    """
    import geopandas as gpd
    shape   = gpd.read_file(shapefile).to_crs('EPSG:4326')
    if hasattr(shape.geometry, 'union_all'):          # new geopandas
        outline = shape.geometry.union_all()
    else:                                             # old geopandas
        outline = shape.geometry.unary_union
    points  = gpd.points_from_xy(stations['lon'], stations['lat'])
    inside  = np.array([outline.covers(pt) for pt in points])
    return stations[inside].reset_index(drop=True)


def select_uniform_stations(stations: pd.DataFrame, n_target: int = 140,
                            lat0: float = 52.0, lon0: float = 10.0):
    """
    Description:
    Farthest-point selection for a uniform spatial coverage:

    Input:
    stations   : DataFrame from load_data_holdings
    n_target   : number of stations to select           [int]
    lat0, lon0 : start point [deg]

    Output:
    DataFrame with the selected stations

    Author : Mohamed Amin and AI.
    Date   : September 2026
    """
    lat = stations['lat'].values
    lon = stations['lon'].values

    first    = np.argmin(haversine_distance(lat0, lon0, lat, lon))
    selected = [first]
    min_dist = haversine_distance(lat[first], lon[first], lat, lon)   # distance to the nearest selected station

    while len(selected) < min(n_target, len(stations)):
        new = np.argmax(min_dist)
        selected.append(new)
        min_dist = np.minimum(min_dist, haversine_distance(lat[new], lon[new], lat, lon))

    return stations.iloc[selected].reset_index(drop=True)


def add_stations(all_stations: pd.DataFrame, selected: pd.DataFrame, names: list):
    """
    Description:
    Add stations to the list

    Input:
    all_stations : DataFrame with all candidate stations
    selected     : DataFrame with the selected stations
    names        : station names to add                 [list]

    Output:
    DataFrame with the selected + added stations

    Author : Mohamed Amin
    Date   : September 2026
    """
    extra = all_stations[all_stations['station'].isin(names) &
                         ~all_stations['station'].isin(selected['station'])]
    return pd.concat([selected, extra]).reset_index(drop=True)


def download_file(url: str, filepath: str, retries: int = 3):
    """
    Description:
    Download one file. Tries again after a short pause if it fails.

    Input:
    url      : file URL                                 [str]
    filepath : where to save it                         [str]
    retries  : number of tries                          [int]

    Output:
    status : 'ok', 'not_found' or 'failed'              [str]

    Author : Mohamed Amin and AI for the retry logic
    Date   : September 2026
    """
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 404:
                return 'not_found'
            r.raise_for_status()
            Path(filepath).write_bytes(r.content)
            return 'ok'
        except requests.RequestException:
            time.sleep(2)
    return 'failed'


def download_stations(names: list, base_url: str, out_dir: str,
                      suffix: str = '.tenv3', pause: float = 0.2):
    """
    Description:
    Download files for every station.
    Files that already exist are skipped.

    Input:
    names    : station names                            [list]
    base_url : folder URL on the NGL server             [str]
    out_dir  : output folder                            [str]
    suffix   : file ending, e.g. '.tenv3' or '.EU.tenv3'
    pause    : wait between downloads [s]

    Output:
    status : dict {station: 'ok' / 'exists' / 'not_found' / 'failed'}

    Author : Mohamed Amin and AI for the retry logic
    Date   : September 2026
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    status = {}
    for k, sta in enumerate(names, 1):
        filepath = out_dir / f'{sta}{suffix}'
        if filepath.exists() and filepath.stat().st_size > 0:
            status[sta] = 'exists'
        else:
            status[sta] = download_file(f'{base_url}/{sta}{suffix}', filepath)
            time.sleep(pause)
        print(f'[{k}/{len(names)}] {sta}: {status[sta]}')
    return status


def write_station_lists(stations: pd.DataFrame, out_dir: str):
    """
    Description:
    Write the station lists

    Input:
    stations : DataFrame with station, lat, lon
    out_dir  : output folder         [str]

    Output:
    three text files in out_dir

    Author : Mohamed Amin
    Date   : September 2026
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stations = stations.sort_values('station').reset_index(drop=True)

    with open(out_dir / 'station_ids.txt', 'w') as f:
        for name in stations['station']:
            f.write(f'{name}\n')

    with open(out_dir / 'station_list.txt', 'w') as f:
        f.write(',0\n')
        for k, name in enumerate(stations['station'], 1):
            f.write(f'{k},{name}\n')

    with open(out_dir / 'station_lat_lon.txt', 'w') as f:
        for _, row in stations.iterrows():
            f.write(f"{row['lat']:.8f} {row['lon']:.8f} {row['station']}\n")
