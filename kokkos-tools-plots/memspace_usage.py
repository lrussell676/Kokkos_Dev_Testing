import os
import matplotlib.pyplot as plt
import argparse

# Directory containing .memspace_usage files
directory = './'

# Parse additional command-line argument for input strings
parser = argparse.ArgumentParser(description='Used to plot memspace_usage data from kokkos-tools. Just set up for Host this now.')
parser.add_argument('--ID', type=str, help='Isolate Process ID')
args = parser.parse_args()

# Additional input string
input_ProcessID = f"{args.ID}-Host.memspace_usage" if args.ID else ".memspace_usage"

# Lists to store data
timestamps = []
values1 = []
values2 = []
values3 = []

# Read data from each .memspace_usage file
for filename in os.listdir(directory):
    if filename.endswith(input_ProcessID):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as file:
            for line in file:
                parts = line.split()
                if len(parts) == 4:
                    timestamps.append(float(parts[0]))
                    values1.append(float(parts[1]))
                    values2.append(float(parts[2]))
                    values3.append(float(parts[3]))

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(timestamps, values1, label='Size(MB)')
plt.plot(timestamps, values2, label='HighWater(MB)')
plt.plot(timestamps, values3, label='HighWater-Process(MB)')
plt.xlabel('Timestamp (s)')
plt.ylabel('Values')
plt.title('Memspace Usage Over Time')
plt.legend()
plt.show()