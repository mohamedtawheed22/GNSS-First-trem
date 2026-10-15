"""
Step 2 : jump catalog for every station (NGL steps file + USGS earthquakes),

Author : Mohamed Amin
Date   : August 2026
"""
import os
import glob
import Read_Wright_GNSS as RWG
import Jump_Catalog     as JC

TENV3_DIR  = r'D:\0_Master_Project\0_Download GNSS Stations\ngl_data\tenv3_IGS14'
STEPS_FILE = r'D:\0_Master_Project\1_Jumps\ngl_steps.txt'
EQ_CSV     = r'D:\0_Master_Project\1_Jumps\global_earthquakes_master.csv'   # None -> NGL steps only
OUT_DIR    = r'D:\0_Master_Project\1_Jumps\jump_catalogs'

MAG_MIN = 5.0       # smallest USGS magnitude
MIN_SEP = 0.03      # merge window [yr] (~11 days)

os.makedirs(OUT_DIR, exist_ok=True)

# 1. catalog for all stations
steps = JC.load_ngl_steps(STEPS_FILE)
eq    = JC.load_usgs_catalog(EQ_CSV) if EQ_CSV else None

for fp in sorted(glob.glob(os.path.join(TENV3_DIR, '*.tenv3'))):
    station = os.path.basename(fp).split('.')[0].upper()
    data    = RWG.load_tenv3(fp)

    jumps = JC.find_jumps(station, data['lat'], data['lon'],
                          data['tp'].min(), data['tp'].max(),
                          steps, eq, MAG_MIN, MIN_SEP)
    JC.save_jumps(os.path.join(OUT_DIR, f'{station}_jumps.json'), station, jumps)

    n_t1 = sum(1 for j in jumps if j['type'] == 1)
    print(f'{station}: {len(jumps)} jumps  (type 1: {n_t1}, type 0: {len(jumps) - n_t1})')

STATION = 'BONN'
data  = RWG.load_tenv3(os.path.join(TENV3_DIR, f'{STATION}.tenv3'))
jumps = JC.load_jumps(os.path.join(OUT_DIR, f'{STATION}_jumps.json'))

