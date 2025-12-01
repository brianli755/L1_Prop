from argparse import ArgumentParser, RawDescriptionHelpFormatter
import pandas as pd
import numpy as np
import pyspedas
import pytz
from pyspedas import tplot
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from datetime import datetime, timedelta
from spacepy.pybats import ImfInput
from matplotlib.lines import Line2D
from numpy import bool_
from numpy.ma import MaskedArray
from scipy.interpolate import interp1d
from matplotlib.dates import date2num
from spacepy.plot import style
from spacepy.datamodel import dmarray
from glob import glob
from os import path
import re
import pytz
from dateutil.parser import isoparse

import functions as f
from pytplot import get_data


parser = ArgumentParser(description=__doc__,
                        formatter_class=RawDescriptionHelpFormatter)
parser.add_argument("input", type=str, help='The list of dates to analyze in the required .CSV format.')
#parser.add_argument("output", type=str, help='The data output file name.')
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
event_list = pd.read_csv(args.input, delimiter=',', header=0)
for e in range(len(event_list)-1):
    start = event_list['Start'][e]
    date_stop = event_list['Stop'][e]

    # Set up OMNI importing code
    sec = 7200
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

    # Finding the correct shift distance
    if args.set_downstream:
        bound_dist = np.average(omni['BSN'])

    # Getting Upstream Data
    # Uses the same stuff as before to generate a new upstream file, also exports a plot of it.
    up_time = []
    shift = []

    bx = []
    by = []
    bz = []

    pt_ID = []
    for i in range(len(omni['IMF'])):
        delta = pd.Timedelta(omni['ts'][i], unit='s')
        time = omni['Time'][i] - delta
        isotime = time.isoformat() #+ '.000Z'
        shift.append(delta)

        match omni['IMF'][i]:
            # If the spacecraft ID is 71, get the data from ace.
            case 71:
                ind = ace.loc[ace['Time'] == omni['Time'].iloc[i] - delta].index[0]
                # if ace doesn't have the data, get the data from wind

                if np.isnan(ace['BZ'][ind]):
                    #("found a nan")
                    ind = wind.loc[wind['Time'] == omni['Time'].iloc[i] - delta].index[0]
                    #print('1')
                    bx.append(wind['BX'][ind])
                    by.append(wind['BY'][ind])
                    bz.append(wind['BZ'][ind])
                    up_time.append(isotime)
                    pt_ID.append(51)
                else:
                    bx.append(ace['BX'][ind])
                    by.append(ace['BY'][ind])
                    bz.append(ace['BZ'][ind])
                    up_time.append(isotime)
                    pt_ID.append(71)

            # if the spacecraft ID is 51 or 52 get the data from wind
            case 51 | 52:
                ind = wind.loc[wind['Time'] == omni['Time'].iloc[i] - delta].index[0]
                # if wind doesn't have the data, get the data from ace
                if np.isnan(wind['BZ'][ind]):
                    #print("3")
                    ind = ace.loc[ace['Time'] == omni['Time'].iloc[i] - delta].index[0]
                    bx.append(ace['BX'][ind])
                    by.append(ace['BY'][ind])
                    bz.append(ace['BZ'][ind])

                    up_time.append(isotime)
                    pt_ID.append(71)

                else:
                    #print('4')
                    bx.append(wind['BX'][ind])
                    by.append(wind['BY'][ind])
                    bz.append(wind['BZ'][ind])

                    up_time.append(isotime)
                    pt_ID.append(51)

            case _:
                id = pt_ID[-1]
                bx.append(np.nan)
                by.append(np.nan)
                bz.append(np.nan)

                up_time.append(isotime)
                pt_ID.append(id)

    mfi = pd.DataFrame({'Time': up_time, 'bx': bx, 'by': by, 'bz': bz})
    mfi = mfi.sort_values(by='Time')
    mfi = mfi.set_index('Time')
    mfi = mfi.reset_index()

    up_time = []
    shift = []

    vx = []
    vy = []
    vz = []

    npr = []
    temp = []

    pt_ID = []
    for i in range(len(omni['IMF'])):
        delta = pd.Timedelta(omni['ts'][i], unit='s')
        time = omni['Time'][i] - delta
        isotime = time.isoformat() #+ '.000Z'
        shift.append(delta)

        match omni['IMF'][i]:
            # If the spacecraft ID is 71, get the data from ace.
            case 71:
                ind = ace2.loc[ace2['Time'] == omni['Time'].iloc[i] - delta].index[0]
                # if ace doesn't have the data, get the data from wind
                if np.isnan(ace2['VX'][ind]):
                    ind = wind2.loc[wind2['Time'] == omni['Time'].iloc[i] - delta].index[0]
                    vx.append(wind2['VX'][ind])
                    vy.append(wind2['VY'][ind])
                    vz.append(wind2['VZ'][ind])

                    npr.append(wind2['NP'][ind])
                    temp.append(wind2['Temp'][ind])

                    pt_ID.append(51)
                    up_time.append(isotime)

                else:
                    vx.append(ace2['VX'][ind])
                    vy.append(ace2['VY'][ind])
                    vz.append(ace2['VZ'][ind])

                    npr.append(ace2['NP'][ind])
                    temp.append(ace2['Temp'][ind])
                    up_time.append(isotime)
                    pt_ID.append(71)

            # if the spacecraft ID is 51 or 52 get the data from wind
            case 51 | 52:
                ind = wind2.loc[wind2['Time'] == omni['Time'].iloc[i] - delta].index[0]
                # if wind doesn't have the data, get the data from ace
                if np.isnan(wind2['VX'][ind]):
                    ind = ace2.loc[ace2['Time'] == omni['Time'].iloc[i] - delta].index[0]
                    vx.append(ace2['VX'][ind])
                    vy.append(ace2['VY'][ind])
                    vz.append(ace2['VZ'][ind])

                    npr.append(ace2['NP'][ind])
                    temp.append(ace2['Temp'][ind])
                    up_time.append(isotime)
                    pt_ID.append(71)
                else:
                    vx.append(wind2['VX'][ind])
                    vy.append(wind2['VY'][ind])
                    vz.append(wind2['VZ'][ind])

                    npr.append(wind2['NP'][ind])
                    temp.append(wind2['Temp'][ind])
                    up_time.append(isotime)
                    pt_ID.append(51)

            case _:
                id = pt_ID[-1]
                vx.append(np.nan)
                vy.append(np.nan)
                vz.append(np.nan)

                npr.append(np.nan)
                temp.append(np.nan)
                up_time.append(isotime)
                pt_ID.append(id)

    swe = pd.DataFrame({'Time': up_time, 'VX': vx, 'VY': vy, 'VZ': vz, 'Temp': temp, 'NP': npr})
    swe = swe.sort_values(by='Time')
    swe = swe.set_index('Time')
    swe = swe.reset_index()

    # Plotting the upstream vs. OMNI
    #fig, ax = plt.subplots(4, 1, figsize = (12, 16))
    #ax[0].plot(omni['Time'], omni['VZ'], label='Omni', color='red', linestyle='-')

    # Begin main propagation script:

    # Create unified time:
    t_swe, t_mag = swe['Time'][:], mfi['Time'][:]
    time = f.unify_time(t_swe, t_mag)
    time = [isoparse(t) for t in time]

    # Convert coordinates:
    vx, vy, vz = f.gse_to_gsm(swe['VX'][:],
                            swe['VY'][:],
                            swe['VZ'][:], swe['Time'][:])

    # Convert temperature
    temp = swe['Temp'][:]

    # Extract plasma parameters
    raw = {'time': time,
           'n': f.pair(t_swe, swe['NP'][:], time, varname='n'),
           't': f.pair(t_swe, temp, time, varname='t'),
           'ux': f.pair(t_swe, vx, time, varname='ux'),
           'uy': f.pair(t_swe, vy, time, varname='uy'),
           'uz': f.pair(t_swe, vz, time, varname='uz')}

    # Extract magnetic field parameters:
    for b in ['bx', 'by', 'bz']:
        raw[b] = f.pair(t_mag, mfi[b][:], raw['time'], varname=b)

    # Optional values: Update with more experience w/ wind...
    if 'PGSM' in mfi:
        raw['pos'] = f.pair(t_mag, mfi['PGSM'][:, 0],
                          raw['time'], varname='pos') * RE

    # Create seconds-from-start time array:
    tsec = np.array([(t - raw['time'][0]).total_seconds() for t in raw['time']])

    # Get S/C distance. If not in file, use approximation.
    if 'pos' in raw:
        print('S/C location found! Using dynamic location.')
        raw['X'] = raw['pos'] - bound_dist
    else:
        print('S/C location NOT found, using static L1 distance.')
        raw['X'] = l1_dist - bound_dist

    # Apply velocity smoothing as required
    print(f'Applying smoothing using a 1 window size.')
    velsmooth = medfilt(raw['ux'], 1)

    print('Using BALLISTIC propagation.')
    shift = raw['X']/velsmooth  # Time shift per point.
    # Apply shift to times.
    tshift = np.array([t1 - timedelta(seconds=t2) for t1, t2 in
                       zip(raw['time'], shift)])

    # Ensure that any points that are "overtaken" (i.e., slow wind overcome by
    # fast wind) are removed. First, locate those points:
    keep = [0]
    discard = []
    lasttime = tshift[0]
    for i in range(1, len(raw['time'])):
        if tshift[i] > lasttime:
            keep.append(i)
            lasttime = tshift[i]
        else:
            discard.append(i)
    print(f'Removing "overtaken" points {len(discard)} of {tsec.size} total.')

    # Create new IMF object and populate with propagated values.
    # Use the information above to throw out overtaken points.
    imfout = ImfInput(f'{date_start.strftime('%Y-%m-%d')}_outfile.dat', load=False, npoints=len(keep))
    for v in swmf_vars:
        imfout[v] = dmarray(raw[v][keep], {'units': units[v]})
    imfout['time'] = tshift[keep]
    #imfout.attrs['header'].append(f'Source data: {file1}\n')

    imfout.attrs['header'].append('Ballistically propagted from L1 to ' +
                                      'upstream BATS-R-US boundary\n')
    imfout.attrs['header'].append('\n')
    imfout.attrs['coor'] = 'GSM'
    imfout.attrs['satxyz'] = [np.mean(raw['X'])/RE, 0, 0]
    imfout.attrs['header'].append(f'File created on {datetime.now()}')

    imfout.write()

    # Plot!
    fig = imfout.quicklook(['by', 'bz', 'n', 't', 'ux'])
    plotvars = ['by', 'bz', 'n', 't', 'ux']
    for ax, v in zip(fig.axes, plotvars):
        c = ax.get_lines()[0].get_color()
        ax.plot(raw['time'], raw[v], '--', c=c, alpha=.5)
        #ax.plot(raw['time'][:][discard], raw[v][...][discard],
                #'.', c='crimson', alpha=.5)
    l1 = Line2D([], [], color='gray', lw=4,
                label='Timeshifted Values')
    l2 = Line2D([], [], color='gray', alpha=.5, linestyle='--', lw=4,
                label='Original Values')
    l3 = Line2D([], [], marker='.', mfc='crimson', linewidth=0, mec='crimson',
                markersize=10, label='Removed Points')
    fig.legend(handles=[l1, l2, l3], loc='upper center', ncol=3)
    fig.subplots_adjust(top=.933)
    fig.savefig(f'{date_start.strftime('%Y-%m-%d')}_outputprop.png')