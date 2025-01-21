import os
import matplotlib.pyplot as plt
import argparse
from memory_events_defs import *

# Directory containing .mem_events files
directory = './'

# Parse additional command-line argument for input string
parser = argparse.ArgumentParser(description='Used to plot mem_events data from kokkos-tools. Just set up for Host this now.')
parser.add_argument('--ID', type=str, help='Isolate Process ID')
parser.add_argument('--Vname', type=str, help='Name of the Kokkos View to plot')
args = parser.parse_args()

# Additional input strings
input_ProcessID = f"{args.ID}.mem_events" if args.ID else ".mem_events"
if not args.Vname:
    raise ValueError("The --Vname argument is required.")
input_Vname = args.Vname

file_path = None
for filename in os.listdir(directory):
    if filename.endswith(input_ProcessID):
        file_path = os.path.join(directory, filename)

if not file_path:
    raise FileNotFoundError(f"No file ending with {input_ProcessID} found in directory {directory}")

# Plot the data
plot_mem_events(file_path, input_Vname)