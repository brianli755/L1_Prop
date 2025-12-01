from argparse import ArgumentParser, RawDescriptionHelpFormatter
import pandas as pd
import numpy as np
import pyspedas
from pyspedas import tplot
import matplotlib.pyplot as plt
import functions
from pytplot import get_data


parser = ArgumentParser(description=__doc__,
                        formatter_class=RawDescriptionHelpFormatter)
parser.add_argument("input", type=str, help='The list of dates to analyze in the required .CSV format.')
parser.add_argument("output", type=str, help='The data output file name.')
# The --set_downstream flag takes True or False. True takes the average BSN distance and propagates the data there.
# False propagates it to the BATS-R-US upstream boundary.
parser.add_argument("--set_downstream", default=True, action='store_true',)
args = parser.parse_args()

# Declare important constants
RE = 6371              # Earth radius in kmeters.
l1_dist = 1495980      # L1 distance in km.
kboltz = 1.380649E-23  # Boltzmann constant, J/K
mp = 1.67262192E-27    # Proton mass in Kg
bound_dist = 32 * RE   # Distance to BATS-R-US upstream boundary from Earth.

# Set propagation variable names and units.
swmf_vars = ['bx', 'by', 'bz', 'ux', 'uy', 'uz', 'n', 't']
units = {v: u for v, u in zip(swmf_vars, 3*['nT']+3*['km/s']+['cm-3', 'K'])}


# Import the OMNI, ACE, and Wind data using the list of times
event_list = pd.read_csv(args.input, header=0)
for e, event in enumerate(event_list):
    start = event_list['Start'][e]
    date_stop = event_list['Stop'][e]

    sec = 5400
    date_start = pd.to_datetime(start)
    shift = pd.Timedelta(sec, unit='s')
    ts = date_start - shift
    shifted_start = ts.strftime('%Y-%m-%d %H:%M:%S')

    pyspedas.projects.omni.data(trange=[start, date_stop], datatype='1min', level='hro2', time_clip=True)
    omni = pd.DataFrame({
        'Time': get_data('BZ_GSM')[0],
        'BY': get_data('BY_GSM')[1],
        'BZ': get_data('BZ_GSM')[1],
        'IMF': get_data('IMF')[1],
        "ts": get_data('Timeshift')[1],
        'VX': get_data('Vx')[1],
        'VY': get_data('Vy')[1],
        'VZ': get_data('Vz')[1],
        'Density': get_data('proton_density')[1],
        'Temp': get_data('T')[1],
        'X': get_data('x')[1],
        'BSN': get_data('BSN_x')[1],
    })

    # The data will then be interpolated to 1 minute intervals
    omni['Time'] = pd.to_datetime(omni['Time'], unit='s')
    omni = omni.set_index('Time')
    omni = omni.resample('min').mean().interpolate(method='linear').ffill().bfill()
    omni = omni.reset_index()


#importing ace mfi files
    pyspedas.projects.ace.mfi(trange=[shifted_start, date_stop], datatype='h3', time_clip=True)
    ace = pd.DataFrame({
        'Time': get_data('BGSM')[0],
        'BX': get_data('BGSM')[1][:, 0],
        'BY': get_data('BGSM')[1][:, 1],
        'BZ': get_data('BGSM')[1][:, 2],

    }, columns=['Time', 'BX', 'BY', 'BZ']).replace(to_replace=[-1.00000E+31], value=np.nan)

    # The data will then be interpolated to 1 second intervals
    ace['Time'] = pd.to_datetime(ace['Time'], unit='s')
    ace = ace.set_index('Time')
    ace = ace.resample('s').mean().interpolate(method='linear').ffill()
    ace = ace.reset_index()


#importing ace swe files
    pyspedas.projects.ace.swe(trange=[shifted_start, date_stop], datatype='h0', time_clip=True)
    ace2 = pd.DataFrame({
        'Time': get_data('V_GSE')[0],
        'VX': get_data('V_GSE')[1][:, 0],
        'VY': get_data('V_GSE')[1][:, 1],
        'VZ': get_data('V_GSE')[1][:, 2],
        "NP": get_data('Np')[1],
        "Temp": get_data('Tpr')[1],

    }, columns=['Time', 'VX', 'VY', 'VZ', 'NP', 'Temp']).replace(to_replace=[-1.00000E+31], value=np.nan)

    # The data will then be interpolated to 1 second intervals
    ace2['Time'] = pd.to_datetime(ace2['Time'], unit='s')
    ace2 = ace2.set_index('Time')
    ace2 = ace2.resample('s').mean().interpolate(method='linear').ffill()
    ace2 = ace2.reset_index()



#importing wind mfi files
    pyspedas.projects.wind.mfi(trange=[shifted_start, date_stop], datatype='h0', time_clip=True)
    wind = pd.DataFrame({
        'Time': get_data('B3GSM')[0],
        'BX': get_data('B3GSM')[1][:, 0],
        'BY': get_data('B3GSM')[1][:, 1],
        'BZ': get_data('B3GSM')[1][:, 2],

    }, columns=['Time', 'BX', 'BY', 'BZ']).replace(to_replace=[-1.00000E+31], value=np.nan)

    # The data will then be interpolated to 1 second intervals
    wind['Time'] = pd.to_datetime(wind['Time'], unit='s')
    wind = wind.set_index('Time')
    wind = wind.resample('s').mean().interpolate(method='linear').ffill()
    wind = wind.reset_index()



#importing wind swe files
    pyspedas.projects.wind.swe(trange=[shifted_start, date_stop], datatype='h1', varnames=[], time_clip=True)
    temp = (mp / (2 * kboltz)) * (get_data('Proton_W_moment')[1] * 1000) ** 2
    wind2 = pd.DataFrame({
        'Time': get_data('Proton_VX_moment')[0],
        'VX': get_data('Proton_VX_moment')[1],
        'VY': get_data('Proton_VY_moment')[1],
        'VZ': get_data('Proton_VZ_moment')[1],
        "NP": get_data('Proton_Np_moment')[1],
        'Temp': temp,

    }, columns=['Time', 'VX', 'VY', 'VZ', 'NP', 'Temp']).replace(to_replace=[-1.00000E+31], value=np.nan)

    # The data will then be interpolated to 1 second intervals
    wind2['Time'] = pd.to_datetime(wind2['Time'], unit='s')
    wind2 = wind2.set_index('Time')
    wind2 = wind2.resample('s').mean().interpolate(method='linear').ffill()
    wind2 = wind2.reset_index()


#finding the correct shift distance.

