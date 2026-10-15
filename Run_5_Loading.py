"""
Step 5 : subtract NTOL + HYDL (detrended) from the residuals and compare with NTAL.

Author : Mohamed Amin
Date   : September 2026
"""
import os
import glob
import numpy as np
import pandas as pd
import Read_Wright_GNSS   as RWG
import Loading_Correction as LC
from Statistical_Test import rms, rms_reduction

RESID_DIR = r'D:\0_Master_Project\3_Trend_Jumps\01_Residual_tenv3'
LOAD_DIR  = r'D:\0_Master_Project\0_Download GNSS Stations\ngl_data\loading_predictions'
OUT_DIR   = r'D:\0_Master_Project\4_Loading'

REMOVE = ['ntol', 'hydl']     # subtracted
SIGNAL = 'ntal'               # only compared

CORR_DIR = os.path.join(OUT_DIR, '01_Corrected_tenv3')
FIG_DIR  = os.path.join(OUT_DIR, '02_Figures')
os.makedirs(CORR_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

#  run all stations
summary = []
for fp in sorted(glob.glob(os.path.join(RESID_DIR, '*.tenv3'))):
    station   = os.path.basename(fp).split('.')[0].upper()
    load_file = os.path.join(LOAD_DIR, f'{station}.tenv3')
    if not os.path.exists(load_file):
        print(f'{station}: no loading file')
        continue
    try:
        data   = RWG.load_tenv3(fp)
        load   = RWG.load_loading(load_file)
        result = LC.subtract_loading(data, load, REMOVE)

        # keep only the common epochs and write the corrected values
        keep = np.zeros(len(data['tp']), dtype=bool)
        keep[result['idx']] = True
        out = RWG.subset_data(data, keep)
        out = RWG.replace_enu(out, result['east'], result['north'], result['up'])
        RWG.save_tenv3(os.path.join(CORR_DIR, f'{station}.tenv3'), out)


        row = {'station': station, 'lat': data['lat'], 'lon': data['lon'],
               'n_matched': len(result['idx']), 'status': 'OK'}
        for comp, c in [('east', 'e'), ('north', 'n'), ('up', 'u')]:
            before = data[comp][result['idx']]
            row[f'rms_{c}_before_mm'] = rms(before) * 1000
            row[f'rms_{c}_after_mm']  = rms(result[comp]) * 1000
            row[f'red_{c}_%']         = rms_reduction(before, result[comp])
            row[f'corr_{c}_{SIGNAL}'] = np.corrcoef(result[comp], result[SIGNAL][comp])[0, 1]
        summary.append(row)
        print(f"{station}: r_up({SIGNAL}) = {row['corr_u_' + SIGNAL]:.3f}")

    except Exception as e:
        summary.append({'station': station, 'status': f'FAILED: {e}'})
        print(f'{station}: FAILED  {e}')

pd.DataFrame(summary).to_csv(os.path.join(OUT_DIR, 'loading_summary.csv'), index=False)
