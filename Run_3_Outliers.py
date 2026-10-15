"""
Step 3 : jump-aware iterative outlier removal (Baarda / Pope) for all stations.

Author : Mohamed Amin
Date   : September 2026
"""
import os
import glob
import numpy as np
import pandas as pd
import Read_Wright_GNSS as RWG
import Jump_Catalog     as JC
import Outlier_removal  as OR

TENV3_DIR = r'D:\0_Master_Project\0_Download GNSS Stations\ngl_data\tenv3_IGS14'
JUMPS_DIR = r'D:\0_Master_Project\1_Jumps\jump_catalogs'
OUT_DIR   = r'D:\0_Master_Project\2_Outliers'

P        = 1
Q        = [1, 2, 4]
OMEGA0   = 2 * np.pi
ALPHA    = 0.05
MAX_ITER = 20

CLEAN_DIR = os.path.join(OUT_DIR, '01_Clean_tenv3')
FIG_DIR   = os.path.join(OUT_DIR, '02_Figures')
os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# %% run all stations
summary = []
for fp in sorted(glob.glob(os.path.join(TENV3_DIR, '*.tenv3'))):
    station = os.path.basename(fp).split('.')[0].upper()
    try:
        data  = RWG.load_tenv3(fp)
        jumps = JC.load_jumps(os.path.join(JUMPS_DIR, f'{station}_jumps.json'))

        outlier, counts = OR.outlier_removal_station(data, jumps, P, OMEGA0, Q, ALPHA, MAX_ITER)

        clean = RWG.subset_data(data, ~outlier)
        RWG.save_tenv3(os.path.join(CLEAN_DIR, f'{station}.tenv3'), clean)
      
        summary.append({'station': station, 'n_raw': len(outlier),
                        'n_clean': int(np.sum(~outlier)), 'n_jumps': len(jumps),
                        **counts, 'status': 'OK'})
        print(f"{station}: removed {counts['total']}/{len(outlier)}")

    except Exception as e:
        summary.append({'station': station, 'status': f'FAILED: {e}'})
        print(f'{station}: FAILED  {e}')

pd.DataFrame(summary).to_csv(os.path.join(OUT_DIR, 'outlier_summary.csv'), index=False)
