## This is my directory for testing KOKKOS Development - README Updated (13/02/24)

Current folders of interest are Ellipsoid_10atoms and Ellipsoid_262Katoms.

I have a few commit messages that are outdated and can be ignored - these are the ones saying I'm having issues on GPU or 256Katoms scale, but I appear so have resolved these (without comm device).

As it stands, everything apart from running "-pk kokkos comm device" appear to be correct, compiling from https://github.com/lrussell676/lammps/commit/b28a29a97a42668ffa7904d8f474c1604370f483 in both Ellipsoid_[10/262K]atoms folders:

Folder names represent running as:
- CPU_np[1/4] - Standard LAMMPS (no KOKKOS package), lmp_mpi runs with "mpirun -np [1/4]"
- kk_g1 - KOKKOS on GPU (CUDA), as in "mpirun -np 1 lmp_exe -in lmp.in -k on g 1 -sf kk"
- kk_g1_t2 - KOKKOS on GPU (CUDA) with OpenMP, as in "mpirun -np 1 lmp_exe -in lmp.in -k on g 1 t 2 -sf kk"
- kk_mpi_np[1/4] - KOKKOS on mpi_only, as in "mpirun -np [1/4] lmp_exe -in lmp.in -k on -sf kk"
- kk_omp_npAtB - KOKKOS on OpenMP, as in "mpirun -np A lmp_exe -in lmp.in -k on t B -sf kk"

Currently havings errors when attempting to run "-pk kokkos comm device", and only currently testing within "comm_device_kk_mpi_np4" folder (KOKKOS on mpi_only and with comm device, as in "mpirun -np 4 lmp_exe -in lmp.in -k on -sf kk -pk kokkos comm device"). Same issues running -np 2.

Both Ellipsoid_*atoms folder contain the lmp.in and lmp_data files I've been using for testing. I should add that in the case of 262K atoms, I've been toggling on/off "reset_atoms", and also atomic data is generated directly from the input script rather than an included lmp_data file.


