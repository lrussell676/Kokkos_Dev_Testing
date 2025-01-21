import matplotlib.pyplot as plt

def plot_mem_events(file_path, input_Vname):
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
            time = float(parts[0])
            size = int(parts[2])
            op = parts[4]
            name = parts[5]
            
            if input_Vname in name:
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
    plt.title(f'Memory Events for {input_Vname}')
    plt.legend()
    plt.show()