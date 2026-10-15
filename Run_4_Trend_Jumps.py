"""
Step 4 : subtract trend + jumps from the clean stations (seasonal signal is kept).

Author : Mohamed Amin
Date   : September 2026
"""

import os
import glob
import numpy as np
import pandas as pd
import Read_Wright_GNSS as RWG
import Jump_Catalog     as JC
import Trend_Jumps      as TJ
from Statistical_Test import rms

CLEAN_DIR = r'D:\0_Master_Project\2_Outliers\01_Clean_tenv3'
JUMPS_DIR = r'D:\0_Master_Project\1_Jumps\jump_catalogs'
OUT_DIR   = r'D:\0_Master_Project\3_Trend_Jumps'

P      = 1              # 1 = linear trend
Q      = [1, 2, 4]      # estimated with the trend but NOT subtracted ([] = no harmonics in the fit)
OMEGA0 = 2 * np.pi

RESID_DIR = os.path.join(OUT_DIR, '01_Residual_tenv3')
FIG_DIR   = os.path.join(OUT_DIR, '02_Figures')
os.makedirs(RESID_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

#  run all stations
summary = []
for fp in sorted(glob.glob(os.path.join(CLEAN_DIR, '*.tenv3'))):
    station = os.path.basename(fp).split('.')[0].upper()
    try:
        data  = RWG.load_tenv3(fp)
        jumps = JC.load_jumps(os.path.join(JUMPS_DIR, f'{station}_jumps.json'))

        resid, model, used = TJ.subtract_trend_jumps(data, jumps, P, OMEGA0, Q)

        out = RWG.replace_enu(data, resid['east'], resid['north'], resid['up'])
        RWG.save_tenv3(os.path.join(RESID_DIR, f'{station}.tenv3'), out)

        summary.append({'station': station, 'n_obs': len(data['tp']),
                        'n_jumps_used': len(used),
                        'rms_e_mm': rms(resid['east'])  * 1000,
                        'rms_n_mm': rms(resid['north']) * 1000,
                        'rms_u_mm': rms(resid['up'])    * 1000,
                        'status': 'OK'})
        print(f"{station}: RMS up = {rms(resid['up']) * 1000:.2f} mm")

    except Exception as e:
        summary.append({'station': station, 'status': f'FAILED: {e}'})
        print(f'{station}: FAILED  {e}')

pd.DataFrame(summary).to_csv(os.path.join(OUT_DIR, 'trend_summary.csv'), index=False)
