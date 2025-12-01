from argparse import ArgumentParser, RawDescriptionHelpFormatter
import pandas as pd
import numpy as np
import pyspedas
from pyspedas import tplot
import matplotlib.pyplot as plt

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


# Import the OMNI, ACE, and Wind data using the list of times
event_list = pd.read_csv(args.input, header=0)
for e, event in enumerate(event_list):
    date_start = event_list['Start'][e]
    date_stop = event_list['Stop'][e]