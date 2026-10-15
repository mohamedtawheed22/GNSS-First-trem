"""
Step 1 : select European GNSS stations and download tenv3 + loading predictions.

Author : Mohamed Amin and Anne Klos 
Date   : September 2026
"""
import os
import Station_Download as SD
import Plot_GNSS        as PG

OUT_DIR   = r'D:\0_Master_Project\0_Download GNSS Stations\ngl_data'
TENV3_URL = 'https://geodesy.unr.edu/gps_timeseries/IGS14/tenv3/IGS14'
LOAD_URL  = 'https://geodesy.unr.edu/gps_timeseries/tenv3_loadpredictions'
SHAPEFILE = None            # e.g. r'D:\RA_2\GNSS\shapefile\NUTS_RG_01M_2024_3035.shp'

N_TARGET  = 140             # number of uniform stations
MUST_HAVE = ['BONN', 'POTS', 'WTZR', 'ZIMM', 'BOR1', 'ONSA',
             'REYK', 'MATE', 'GRAS', 'HERS', 'BRUS', 'WSRT']

#  1. select the stations
all_stations = SD.load_data_holdings()

if SHAPEFILE is None:
    # uniform selection inside a box, at least 5 years of data
    europe   = SD.stations_in_box(all_stations, 35, 72, -25, 45, min_solutions=1825)
    selected = SD.select_uniform_stations(europe, N_TARGET)
    selected = SD.add_stations(europe, selected, MUST_HAVE)
else:
    # all stations inside the shapefile
    selected = SD.stations_in_shapefile(all_stations, SHAPEFILE)

print(f'{len(selected)} stations selected')
SD.write_station_lists(selected, OUT_DIR)
PG.plot_station_map(selected, out_png=os.path.join(OUT_DIR, 'station_map.png'))

#  2. download
names = list(selected['station'])
SD.download_stations(names, TENV3_URL, os.path.join(OUT_DIR, 'tenv3_IGS14'))
SD.download_stations(names, LOAD_URL,  os.path.join(OUT_DIR, 'loading_predictions'))
