To run on Summit:

Load modules:
module load pgi/19.9
module load cuda/10.1.243

PGI is used because I already had a working version of Kokkos on Summit installed with PGI.

Compiling is the same as before, e.g.:

make DEVICE=GPU

Changes:
1. DEVICE=CPU uses OpenMP; to disable, use DEVICE=SERIAL
2. ADADELTA is the ONLY option right now
3. NWI has changed to NUM_OF_THREADS_PER_BLOCK. Specify as desired; the default is 32 on GPU and 1 on CPU
4. OVERLAP=ON is now an option if DEVICE=GPU. This will create 2 OpenMP threads, with 1 handling the setup of the next file.
5. -filelist example.txt is now an option. This overwrites any -lfile and -ffile input. If the filelist is present, the program will step through the list one by one. If OVERLAP is on, the setup of the (i+1)th file will occur while the (i)th file is executing on GPU. An example filelist might look like this:
./input/1ac8/derived/1ac8_protein.maps.fld
./input/1ac8/derived/1ac8_ligand.pdbqt
./input/1stp/derived/1stp_protein.maps.fld
./input/1stp/derived/1stp_ligand.pdbqt
In that case, the *.dlg outputs will be whatever is supplied by the -resnam option (or the default), plus the loop index. i.e. docking0.dlg, docking1.dlg. Alternatively, The desired output names can be supplied in the filelist:
1ac8_results
./input/1ac8/derived/1ac8_protein.maps.fld
./input/1ac8/derived/1ac8_ligand.pdbqt
1stp_results
./input/1stp/derived/1stp_protein.maps.fld
./input/1stp/derived/1stp_ligand.pdbqt
This will write the results to 1ac8_results.dlg, 1stp_results.dlg. A mixed format (names only for some) are not supported. The order is not important, as long as there are the same number of each (i.e. you could put 6 *.fld files, then 6 *.pdbqt files if thats more convenient.) For best performance, you would want large setups to overlap with large executions, so sort the files by anticipated size before launching.

The new option is back-compatible, i.e. it will still run normally with a single job provided with -lfile and -ffile.

A sample Summit script is included in the repository: sample_summit_jobscript
