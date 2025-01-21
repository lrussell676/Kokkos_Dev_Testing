import os
import argparse
from memory_events_defs import *

# Directory containing .mem_events files
directory = './'

# Parse additional command-line argument for input string
parser = argparse.ArgumentParser(description='Used to process mem_events data from kokkos-tools. Just set up for Host this now.')
# Mandatory arguments
parser.add_argument('--mode', type=str, choices=['c', 'p'], required=True, help='Mode of operation: "c" for check, "p" for plot. (mandatory)')
parser.add_argument('--Vname', type=str, help='Name of the Kokkos View to plot. (mandatory for "p" plot mode, not used for "c" check mode)')
# Optional arguments
parser.add_argument('--ID', type=str, help='Isolate Process ID file (optional)')
args = parser.parse_args()

# Input strings
# -------------------------------------------------------------------------------------------------------------------------------
# --Vname
if args.mode == 'p':
    if not args.Vname:
        raise ValueError("The --Vname argument is required for plot mode.")
    input_Vname = args.Vname
# --ID
input_ProcessID = f"{args.ID}.mem_events" if args.ID else ".mem_events"
file_paths = []

for filename in os.listdir(directory):
    if filename.endswith(input_ProcessID):
        file_paths.append(os.path.join(directory, filename))

if not file_paths and input_ProcessID:
    raise FileNotFoundError(f"No file ending with {input_ProcessID} found in directory {directory}")

# Plot Mode
# -------------------------------------------------------------------------------------------------------------------------------
if args.mode == 'p':
    for file_path in file_paths:
        plot_mem_events(file_path, input_Vname)

# Check Mode
# -------------------------------------------------------------------------------------------------------------------------------
if args.mode == 'c':
    for file_path in file_paths:
        check_mem_events(file_path)