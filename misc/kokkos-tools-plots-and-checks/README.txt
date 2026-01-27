# Overview of the .py files.

This folder is designed to have kokkos-tools output files copied into here ("./").

# memory_events.py
`memory_events.py` is useful for extracting all the info from mem_events output. 

Just set up for Host this now.

options:
  --mode {c,p}   Mode of operation: "c" for check, "p" for plot. (mandatory)
  --Vname VNAME  Name of the Kokkos View to plot. (mandatory for "p" plot mode, not used for "c" check mode)
  --ID ID        Isolate Process ID file. (optional)

`c` Check mode is the most useful. It scans through all the Kokkos View Names in the file and issues warnings
anytime the size of memory allocation and deallocation is not 0 by the end of the file.

`p` Plot mode plots a graph of memory allocation and deallocation over time for a given Kokkos View.

examples:
- python memory_events.py --mode c
- python memory_events.py --mode c --ID 8431 >stdout
- python memory_events.py --mode p --Vname DualView::modified_flags
- python memory_events.py --mode p --ID 8431 --Vname "UnorderedMap - size"

# memspace_usage.py
`memspace_usage.py` just plots a graph of the memspace_usage output.

Just set up for Host this now.

Best to provide one file at a time (--ID CLI arg), otherwise the plots look odd.

examples:
- python memspace_usage.py --ID 8432