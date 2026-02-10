from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import os
import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)

#------ Define functions here --------
def find_max_value_and_index(values):
    if all(c < 0 for c in values[1:]):
        return max(values[1:], key=abs), values.index(max(values[1:], key=abs))
    else:
        return max(values[1:]), values.index(max(values[1:]))

def correlate_no_pe_vectorized(file_pair, output_path, variable='BZ_GSM'):
    artemis, omni = file_pair
    num_windows = len(omni) - 60

    # Pre-allocate arrays
    data_rows = []
    offset_rows = []

    # Process windows in batches
    for n in range(num_windows):
        artemis_start = artemis.loc[artemis['Time'] == omni['Time'].iloc[n]].index[0]
        artemis_stop = artemis.loc[artemis['Time'] == omni['Time'].iloc[n + 59]].index[0]

        # Position calculations
        o_avg_xpos = np.average(omni['XPOS'][n:n + 59])
        a_avg_xpos = np.average(artemis['XPOS'][artemis_start:artemis_stop])
        hourly_offset = a_avg_xpos - o_avg_xpos
        hourly_velocity = np.average(omni['VX'][n:n + 59])
        pred_arrival = int((hourly_offset / np.abs(hourly_velocity)) / 60)

        # Create array of all shifts at once
        o_slice = omni[variable][n:n + 59].values
        a_shifts = np.array([artemis[variable][artemis_start - i:artemis_stop - i].values
                           for i in range(31)])

        # Calculate all correlations at once using np.corrcoef
        correlations = np.array([np.corrcoef(o_slice, a_shift)[0,1] for a_shift in a_shifts])

        pearson_max_value = np.max(correlations)
        pearson_max_index = np.argmax(correlations)

        data_rows.append([omni['Time'][n], omni['Time'][n + 59], pearson_max_value,
                         hourly_offset, hourly_velocity, pred_arrival])
        offset_rows.append([omni['Time'][n], omni['Time'][n + 60], pearson_max_index,
                          pred_arrival])

    # Create DataFrames and save results
    values = pd.DataFrame(data_rows, columns=['Start', 'Stop', 'Pearson', 'hourly-position',
                                            'hourly-velocity', 'expected-arrival'])
    shifts = pd.DataFrame(offset_rows, columns=['Start', 'Stop', 'Pearson', 'expected-arrival'])

    # Create directories if they don't exist
    os.makedirs(os.path.join(output_path, f'{variable}/metrics/'), exist_ok=True)
    os.makedirs(os.path.join(output_path, f'{variable}/shifts/'), exist_ok=True)

    # Save results
    values.to_csv(os.path.join(output_path, f'{variable}/metrics/{artemis["Time"][0].strftime("%Y-%m-%d")}.csv'))
    shifts.to_csv(os.path.join(output_path, f'{variable}/shifts/{artemis["Time"][0].strftime("%Y-%m-%d")}.csv'))
#-------------------------------------

#----- Code that does the correlation part -------------
import warnings
from doCorrelate import *
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.makedirs('../High Speed Streams/correlations/', exist_ok=True)

omniFileList = sorted(filter(lambda x: os.path.isfile(os.path.join('../High Speed Streams/processed_solar_wind/omni', x)), os.listdir('../High Speed Streams/processed_solar_wind/omni')))
artemisFileList = sorted(filter(lambda x: os.path.isfile(os.path.join('../High Speed Streams/processed_solar_wind/artemis', x)), os.listdir('../High Speed Streams/processed_solar_wind/artemis')))
omni_dict = {omni_file.replace('omni_', ''): os.path.join('../High Speed Streams/processed_solar_wind/omni', omni_file)
             for omni_file in omniFileList
             if not omni_file.startswith('.')}
file_pairs = []
for artemis_file in artemisFileList:
    if artemis_file.startswith('.'):
        continue
    # Get the date part by removing 'artemis_' prefix
    date_part = artemis_file.replace('artemis_', '')
    # If we have matching files, create the triplet
    if date_part in omni_dict:
        artemis_path = os.path.join('../High Speed Streams/processed_solar_wind/artemis', artemis_file)
        file_pairs.append((artemis_path, omni_dict[date_part]))

for var in ['BX_GSM','BY_GSM','BZ_GSM','VX','N','T']:
#for var in ['BZ_GSM']:
    for f in file_pairs:
        print(f)
        artemis_file = pd.read_csv(f[0], delimiter=',', header=0)
        omni_file = pd.read_csv(f[1], delimiter=',', header=0)
        #artemis_file = artemis_file.rename(columns={'XPOS': 'Xpos'}) # FIXED IN GetSatellitesGSM!!!
        # Reformat the time column to DateTime objects../Ordinary Solar Wind/omni'
        artemis_file['Time'] = pd.to_datetime(artemis_file['Time'], format='%Y-%m-%d %H:%M:%S')
        omni_file['Time'] = pd.to_datetime(omni_file['Time'], format='%Y-%m-%d %H:%M:%S')
        artemis_file['V'] = np.sqrt(artemis_file['VX']**2 + artemis_file['VY']**2 + artemis_file['VZ']**2)
        omni_file['V'] = np.sqrt(omni_file['VX']**2 + omni_file['VY']**2 + omni_file['VZ']**2)

        print(artemis_file['Time'][0], omni_file['Time'][0])
        correlate_no_pe_vectorized((artemis_file, omni_file), '../High Speed Streams/correlations/', variable=var)

for var in ['BX_GSM','BY_GSM','BZ_GSM','VX','N','T']:
    directory = f'../High Speed Streams/correlations/{var}/metrics'
    frame = []
    # Sort filenames in date order
    filenames = sorted(f for f in os.listdir(directory) if not f.startswith('.'))

    # Append files in sorted order
    for filename in filenames:
        file_path = os.path.join(directory, filename)
        file = pd.read_csv(file_path, delimiter=',', header=0, index_col=0)
        frame.append(file)

    # Concatenate into a single DataFrame
    df = pd.concat(frame, axis=0, ignore_index=True).reset_index(drop=True)

    # Save to CSV
    output_dir = f'../High Speed Streams/correlations/{var}/merged'
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, 'output.csv'))