import matplotlib.pyplot as plt
import warnings

############################################################################################################
# Plots memory allocation and deallocation events along with cumulative memory usage over time.
############################################################################################################
def plotting_memspace_usage(times_allo, sizes_allocate, times_deallo, 
                            sizes_deallocate, times_total, cumulative_sizes, input_Vname):
    plt.figure(figsize=(10, 6))
    plt.scatter(times_allo, sizes_allocate, color='blue', label='Allocate', marker='x')
    plt.scatter(times_deallo, sizes_deallocate, color='red', label='Deallocate', marker='o')
    plt.plot(times_total, cumulative_sizes, color='green', label='Cumulative Sum')
    ymax = max(sizes_allocate)
    ymin = min(sizes_deallocate)
    for i, size in enumerate(sizes_allocate):
        plt.text(times_allo[i], sizes_allocate[i] + 0.03*ymax, f'+{size}', 
             fontsize=8, verticalalignment='bottom', horizontalalignment='center')
    for i, size in enumerate(sizes_deallocate):
        plt.text(times_deallo[i], sizes_deallocate[i] + 0.04*ymin, f'-{size}', 
                 fontsize=8, verticalalignment='top', horizontalalignment='center')
    if cumulative_sizes:
        plt.text(times_total[-1], cumulative_sizes[-1], f'{cumulative_sizes[-1]}', 
                 fontsize=9, verticalalignment='bottom', horizontalalignment='center')
    plt.xlabel('Time (s)')
    plt.ylabel('Size (bytes)')
    plt.title(f'Memory Events for "{input_Vname}"')
    plt.legend()
    plt.show()

############################################################################################################
# Reads memory events from a file, filters them by a given variable name, and plots the events.
############################################################################################################
def check_and_plot_mem_event(file_path, input_Vname):
    print(f"Plotting memory events for {input_Vname} from file {file_path}")
    times_allo = []
    times_deallo = []
    times_total = []
    sizes_allocate = []
    sizes_deallocate = []
    cumulative_sizes = []
    cumulative_sum = 0

    found_vname = False
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            time = float(parts[0])
            size = int(parts[2])
            op = parts[4]
            name = ' '.join(parts[5:])
            
            if input_Vname in name:
                found_vname = True
                if op == "Allocate":
                    times_allo.append(time)
                    sizes_allocate.append(size)
                    #sizes_deallocate.append(0)
                    cumulative_sum += size
                elif op == "DeAllocate":
                    times_deallo.append(time)
                    sizes_deallocate.append(size)
                    #sizes_allocate.append(0)
                    cumulative_sum += size
                times_total.append(time)
                cumulative_sizes.append(cumulative_sum)
    
    if not found_vname:
        raise ValueError(f"Input Vname ''{input_Vname}'' not found in file {file_path}. Check the Vname argument.")

    plotting_memspace_usage(times_allo, sizes_allocate, times_deallo, 
                            sizes_deallocate, times_total, cumulative_sizes, input_Vname)
    
############################################################################################################
# Reads memory events from a file, checks for unique variable names, and verifies if memory events are 
# balanced for each name.
############################################################################################################
def check_mem_events(file_path):
    print(f"Checking memory event names in file {file_path}")
    names = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            name = ' '.join(parts[5:])
            if name not in names:
                names.append(name)
    print(f"Unique Names found in file: {len(names)}")

    warning_count = 0
    warning_names = []

    for name in names:
        print("Checking memory events for name: ", name)
        times_allo = []
        times_deallo = []
        times_total = []
        sizes_allocate = []
        sizes_deallocate = []
        cumulative_sizes = []
        cumulative_sum = 0

        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                name_redraw = ' '.join(parts[5:])
                if name != name_redraw:
                    continue
                time = float(parts[0])
                size = int(parts[2])
                op = parts[4]
            
                if op == "Allocate":
                    times_allo.append(time)
                    sizes_allocate.append(size)
                    cumulative_sum += size
                elif op == "DeAllocate":
                    times_deallo.append(time)
                    sizes_deallocate.append(size)
                    cumulative_sum += size
                times_total.append(time)
                cumulative_sizes.append(cumulative_sum)
    
            if cumulative_sizes[-1] != 0:
                warnings.warn(f"\n!!!\nDifference between total allocated and deallocated memory for {name_redraw} is {cumulative_sizes[-1]}\n!!!", RuntimeWarning)
                warning_count += 1
                warning_names.append(name_redraw)
            if cumulative_sizes[-1] != 0:
                print("/ ----------------------------------------------------------------- /")
                print(f"Memory events for {name_redraw} are not balanced. Values are:\n"
                      f"Times Allocated:...... {len(times_allo)} \n"
                      f"Sizes Allocated:...... {sizes_allocate} \n"
                      f"Times Deallocated:.... {len(times_deallo)} \n"
                      f"Sizes Deallocated:.... {sizes_deallocate} \n"
                      f"Total Cumulative Size: {cumulative_sizes}")
                print("/ ----------------------------------------------------------------- /")
    
    print("#\n#\n#")
    print("### Summary ###")
    print("For file: ", file_path)
    print(f"Total number of warnings: {warning_count}")
    if warning_names:
        print("Names with warnings:")
        for name in warning_names:
            print(name)
    print("### End of File Check ###")