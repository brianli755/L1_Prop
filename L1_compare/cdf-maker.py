'''
Takes solar wind .csv files from ACE and converts them to .cdf files to use with the l1_propagate.py script

The code is not working currently. It takes two identical ACE files, specified in the code, and supposedly combines them
There is currently a big error.

Use at your own risk. Or fix and do a pull request. IDC.
'''

import pandas as pd
import numpy as np
from spacepy import pycdf

print(f'Welcome to CDF maker :)')
print(f'Now making your CDF! ...')

# Input CSV filenames
mfi_csv = 'AC_H0_MFI_3606118.csv'   # magnetic field
swe_csv = 'AC_H0_SWE_3606118.csv'   # plasma
out_cdf = 'ACE_combined_3606118.cdf'

# --- Load data --- #
mfi = pd.read_csv(mfi_csv)
swe = pd.read_csv(swe_csv)

# Normalize column names
mfi.columns = mfi.columns.str.strip()
swe.columns = swe.columns.str.strip()

# Parse timestamps (slightly different column names)
mfi['EPOCH'] = pd.to_datetime(mfi[mfi.columns[0]])
swe['EPOCH'] = pd.to_datetime(swe[swe.columns[0]])

# Interpolate SWE (plasma) to MFI (magnetic) timestamps for sync
merged = pd.DataFrame({'Epoch': mfi['EPOCH']})
merged['BX'] = np.interp(
    mfi['EPOCH'].astype(np.int64),
    mfi['EPOCH'].astype(np.int64),
    mfi.iloc[:, 1].values
)
merged['BY'] = np.interp(
    mfi['EPOCH'].astype(np.int64),
    mfi['EPOCH'].astype(np.int64),
    mfi.iloc[:, 2].values
)
merged['BZ'] = np.interp(
    mfi['EPOCH'].astype(np.int64),
    mfi['EPOCH'].astype(np.int64),
    mfi.iloc[:, 3].values
)

# Match plasma data by nearest time
swe_interp = swe.set_index('EPOCH').reindex(merged['Epoch'], method='nearest')

# Add plasma data
merged['VX'] = swe_interp.iloc[:, 4].values
merged['VY'] = swe_interp.iloc[:, 5].values
merged['VZ'] = swe_interp.iloc[:, 6].values
merged['Np'] = swe_interp.iloc[:, 1].values
merged['TEMP'] = swe_interp.iloc[:, 2].values
merged['alpha_ratio'] = swe_interp.iloc[:, 3].values

# Spacecraft position from MFI
merged['SC_pos_GSM_X'] = mfi.iloc[:, 4].values
merged['SC_pos_GSM_Y'] = mfi.iloc[:, 5].values
merged['SC_pos_GSM_Z'] = mfi.iloc[:, 6].values

print(f"Merged dataset shape: {merged.shape}")

# --- Write to CDF --- #
with pycdf.CDF(out_cdf, '') as cdf:
    cdf['Epoch'] = merged['Epoch'].to_numpy(dtype='datetime64[ns]')
    cdf['BGSM'] = merged[['BX', 'BY', 'BZ']].to_numpy()
    cdf['V_GSM'] = merged[['VX', 'VY', 'VZ']].to_numpy()
    cdf['Np'] = merged['Np'].to_numpy()
    cdf['Tpr'] = merged['TEMP'].to_numpy()
    cdf['alpha_ratio'] = merged['alpha_ratio'].to_numpy()
    cdf['SC_pos_GSM'] = merged[['SC_pos_GSM_X', 'SC_pos_GSM_Y', 'SC_pos_GSM_Z']].to_numpy()

    # Metadata
    cdf.attrs['Source'] = 'Combined from ACE CSV MFI + SWE'
    cdf.attrs['Instrument'] = 'ACE MAG + SWE'
    cdf.attrs['Created_By'] = 'CSV-to-CDF converter script'

print(f"Created {out_cdf}")


