import pandas as pd

from glob import glob
from os import path
import re
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from datetime import datetime, timedelta

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import medfilt

from matplotlib.lines import Line2D

from spacepy.pycdf import CDF
from spacepy.plot import style
from spacepy.datamodel import dmarray
from spacepy.pybats import ImfInput


# Declare important constants
RE = 6371              # Earth radius in kmeters.
l1_dist = 1495980      # L1 distance in km.
kboltz = 1.380649E-23  # Boltzmann constant, J/K
mp = 1.67262192E-27    # Proton mass in Kg
bound_dist = 32 * RE   # Distance to BATS-R-US upstream boundary from Earth.

# Set var names and units.
swmf_vars = ['bx', 'by', 'bz', 'ux', 'uy', 'uz', 'n', 't']
units = {v: u for v, u in zip(swmf_vars, 3*['nT']+3*['km/s']+['cm-3', 'K'])}


def gse_to_gsm(x, y, z, time):
    '''
    Use spacepy's Coord module to rotate from GSE to GSM.

    Parameters
    ----------
    x, y, z : Numpy array-like
        The GSE X, Y, and Z values of the timeseries to rotate.
    time : array of Datetimes
        The time corresponding to the xyz timeseries.

    Returns
    -------
    x, y, z : Numpy arrays
        The rotated values now in GSM coordinates.
    '''

    from spacepy.coordinates import Coords
    from spacepy.time import Ticktock
    print(type(time[1]))
    print('~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(time[0])
    print(time[1])
    # Convert time to ticktocks:
    ticks = Ticktock(time,'ISO')


    # Rearrange values to correct shape:
    xyz = np.array([x, y, z]).transpose()

    # Rotate:
    gse_vals = Coords(xyz, 'GSE', 'car', ticks=ticks)
    gsm_vals = gse_vals.convert('GSM', 'car')

    return gsm_vals.x, gsm_vals.y, gsm_vals.z


def unify_time(time1, time2):
    '''
    Given two timeseries, combine all unique points into a single array.
    '''

    time1 = time1.tolist()
    time2 = time2.tolist()

    for t in time2:
        if t not in time1:
            time1.append(t)

    time1.sort()
    return np.array(time1)


def pair(time1, data, time2, varname=None, **kwargs):
    '''
    Use linear interpolation to pair two timeseries of data.  Data set 1
    (data) with time t1 will be interpolated to match time set 2 (t2).
    The returned values, d3, will be data set 1 at time 2.
    No extrapolation will be done; t2's boundaries should encompass those of
    t1.

    Bad data values will be removed prior to interpolation. The `varname`
    kwarg will determine what is considered "bad". Possible varnames include:
    n, t, u[xyz], b[xyz], pos

    This function will correctly handle masked functions such that masked
    values will not be considered.

    **kwargs** will be handed to scipy.interpolate.interp1d
    A common option is to set fill_value='extrapolate' to prevent
    bounds errors.
    '''

    from numpy import bool_
    from numpy.ma import MaskedArray
    from scipy.interpolate import interp1d
    from matplotlib.dates import date2num

    # Dates to floats:
    t1 = date2num(time1)
    t2 = date2num(time2)

    # Search for bad data values and remove:
    loc = ~np.isfinite(data) | (np.abs(data) >= 1E15)
    if varname in ['n', 'alpha']:
        loc = (loc) | (data > 1E4)
    elif varname == 't':
        loc = (loc) | (data > 1E7)
    elif varname in ['ux', 'uy', 'uz']:
        loc = (loc) | (np.abs(data) > 1E4)

    t1, data = t1[~loc], data[~loc]

    # Remove masked values (if given):
    if type(data) is MaskedArray:
        if type(data.mask) is not bool_:
            d = data[~data.mask]
            t1 = t1[~data.mask]
        else:
            d = data
    else:
        d = data

    # Create interpolator function
    func = interp1d(t1, d, fill_value='extrapolate', **kwargs)

    # Interpolate and return.
    return func(t2)

# This function spoofs the variables for WIND

def read_wind(file1, file2):
    print(list(file1.columns.values))
    '''
    Given 1 or 2 of 2 Wind data CDFs (one for SWEPAM, one for MAG), load the
    data and map to SWMF variables. If only one file name is given, the
    other will be automatically determined based on the first.

    Return a dictionary where each key is an SWMF var name mapped to the
    corresponding value in the ACE data.
    '''


    # Find partner files; load CDFs.
    if 'swe' or 'SWE' in file1:
        swe = file1
        mag = file2
        print('the type for time is {}'.format(type(mag['Time'][1])))
        print('the type for VX is {}'.format(type(swe['Proton_VX_moment'][1])))
    elif 'mfi' or 'MFI' in file1:
        mag = file1
        swe = file2
    else:
        raise ValueError('Expected "mfi" or "swe" in file name.')

    # Create unified time:
    t_swe, t_mag = swe['Time'][:], mag['Time'][:]
    time = unify_time(t_swe, t_mag)


    # Convert coordinates:
    vx, vy, vz = gse_to_gsm(swe['Proton_VX_moment'][:],
                            swe['Proton_VY_moment'][:],
                            swe['Proton_VZ_moment'][:], swe['Time'][:])

    #vx, vy, vz = swe['Proton_VX_moment'][:], swe['Proton_VY_moment'][:], swe['Proton_VZ_moment'][:]

    # Convert temperature
    #temp = (mp/(2*kboltz))*(swe['Proton_W_moment'][:]*1000)**2

    # Extract plasma parameters
    raw = {'time': time,
           'n': pair(t_swe, swe['Proton_Np_moment'][:], time, varname='n'),
           't': pair(t_swe, swe['Proton_W_moment'][:], time, varname='t'),

           'ux': pair(t_swe, vx, time, varname='ux'),
           'uy': pair(t_swe, vy, time, varname='uy'),
           'uz': pair(t_swe, vz, time, varname='uz')}

    # Extract magnetic field parameters:
    #for i, b in enumerate(['bx', 'by', 'bz']):
        #raw[b] = pair(t_mag, mag['BGSM'][:, i], raw['time'], varname=b)

    for b in ['bx', 'by', 'bz']:
        raw[b] = pair(t_mag, mag[b][:], raw['time'], varname=b)

    # Optional values: Update with more experience w/ wind...
    # if 'alpha_ratio' in swe:
    #     raw['alpha'] = swe['alpha_ratio'][...]
    if 'PGSM' in mag:
        raw['pos'] = pair(t_mag, mag['PGSM'][:, 0],
                          raw['time'], varname='pos') * RE

    return raw


# ## Begin main script ## #
# Look at filename; determine source and convert data.
file1 = pd.read_csv('../test_csv/swe_data.csv', delimiter=',', header=0, parse_dates=['Time'])
file2 = pd.read_csv('../test_csv/mfi_data.csv', delimiter=',', header=0, parse_dates=['Time'])


raw = read_wind(file1, file2)
print(list(file1.columns.values))

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

# Shift time: distance/velocity = timeshift (negative in GSM coords)

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
for i in range(1, raw['time'].size):
    if tshift[i] > lasttime:
        keep.append(i)
        lasttime = tshift[i]
    else:
        discard.append(i)
print(f'Removing "overtaken" points {len(discard)} of {tsec.size} total.')

# Create new IMF object and populate with propagated values.
# Use the information above to throw out overtaken points.
imfout = ImfInput('outfile.dat', load=False, npoints=len(keep))
for v in swmf_vars:
    imfout[v] = dmarray(raw[v][keep], {'units': units[v]})
imfout['time'] = tshift[keep]
imfout.attrs['header'].append(f'Source data: {file1}\n')

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
    ax.plot(raw['time'][...][discard], raw[v][...][discard],
            '.', c='crimson', alpha=.5)
l1 = Line2D([], [], color='gray', lw=4,
            label='Timeshifted Values')
l2 = Line2D([], [], color='gray', alpha=.5, linestyle='--', lw=4,
            label='Original Values')
l3 = Line2D([], [], marker='.', mfc='crimson', linewidth=0, mec='crimson',
            markersize=10, label='Removed Points')
fig.legend(handles=[l1, l2, l3], loc='upper center', ncol=3)
fig.subplots_adjust(top=.933)
fig.savefig('output_prop_info.png')

