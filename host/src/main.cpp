/*

AutoDock-GPU, an OpenCL implementation of AutoDock 4.2 running a Lamarckian Genetic Algorithm
Copyright (C) 2017 TU Darmstadt, Embedded Systems and Applications Group, Germany. All rights reserved.
For some of the code, Copyright (C) 2019 Computational Structural Biology Center, the Scripps Research Institute.

AutoDock is a Trade Mark of the Scripps Research Institute.

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

*/




#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef USE_OMP
#include <omp.h>
#endif
#include <vector>
#include <Kokkos_Core.hpp>
#include "kokkos_settings.hpp"

#include "processgrid.h"
//include "processresult.h"
#include "processligand.h"
#include "getparameters.h"
#include "performdocking.h"
#include "filelist.hpp"

#ifndef _WIN32
// ------------------------
// Time measurement
#include <sys/time.h>
// ------------------------

inline double seconds_since(timeval& time_start)
{
	timeval time_end;
	gettimeofday(&time_end,NULL);
        double num_sec     = time_end.tv_sec  - time_start.tv_sec;
        double num_usec    = time_end.tv_usec - time_start.tv_usec;
        return (num_sec + (num_usec/1000000));
}
#endif

int main(int argc, char* argv[])
{
	Kokkos::initialize();
	{ // Add a scope so that Kokkos views in main are deallocated before reaching Kokkos::finalize()

	FileList filelist;
	int n_files = 1; // default
	bool overlap = false; // default
	int setup_thread = 0; // thread the performs the setup
	int execution_thread = 0; // thread that performs the execution
	int err = 0;
#ifndef _WIN32
	// Start full timer
	timeval time_start, loop_time_start;
	gettimeofday(&time_start,NULL);
	double exec_time, setup_time;
	double total_savings=0;
#endif

	// Objects that are arguments of docking_with_gpu
	// These must each have 2
	Dockpars   mypars[2];
	Liganddata myligand_init[2];
	Gridinfo   mygrid[2];
	Liganddata myxrayligand[2];
	Kokkos::View<float*,HostType> floatgrids[2];
	floatgrids[0] = Kokkos::View<float*,HostType>("floatgrids0", 0);
	floatgrids[1] = Kokkos::View<float*,HostType>("floatgrids1", 0);

	// Read all the file names if -filelist option is on
	if (get_filelist(&argc, argv, filelist) != 0)
		      return 1;

	if (filelist.used){
		n_files = filelist.nfiles;
		printf("\nRunning %d jobs in pipeline mode ", n_files);
#if defined(USE_GPU) && defined(USE_OMP)
		overlap=true; // Not set up to work with nested OpenMP (yet)
		execution_thread = 1; // Assign Thread 1 to do the execution
#endif
	}

	// Print version info
	printf("\nAutoDock-GPU version: %s\n", VERSION);
#ifdef USE_GPU
	printf("Using the GPU version. NUM_OF_THREADS_PER_BLOCK = %d ", NUM_OF_THREADS_PER_BLOCK);
#else
#ifdef USE_OMP
	printf("Using the CPU version with OpenMP. NUM_OF_THREADS_PER_BLOCK = %d ", NUM_OF_THREADS_PER_BLOCK);
#else
	printf("Using the CPU version without OpenMP (serial).");
#endif
#endif

	for (int i_file=0;i_file<(n_files+1);i_file++){ // one extra iteration since its a pipeline
		int s_id = i_file % 2;    // Alternate which set is undergoing setup (s_id)
		int r_id = (i_file+1) %2; // and which is being used in the run (r_id)
		if (i_file<n_files && filelist.used) {
			printf("\n\n-------------------------------------------------------------------");
			printf("\nJob #%d: ", i_file);
			printf("\n   Fields from: %s",  filelist.fld_files[i_file].c_str());
			printf("\n   Ligands from: %s", filelist.ligand_files[i_file].c_str()); fflush(stdout);
		}
#ifndef _WIN32
		// Time measurement: start of loop
		gettimeofday(&loop_time_start,NULL);
#endif
		// Branch into two threads
		//   setup_thread reads files and prepares the inputs to docking_with_gpu
		//   execution_thread runs docking_with_gpu
#if defined(USE_GPU) && defined(USE_OMP)
		#pragma omp parallel
		{
			int thread_id = omp_get_thread_num();
#else
		{
			int thread_id = 0;
#endif
			// Thread 0 does the setup, unless its the last run (so nothing left to load)
			if ((thread_id==setup_thread) && i_file<n_files) {
				//------------------------------------------------------------
				// Capturing names of grid parameter file and ligand pdbqt file
				//------------------------------------------------------------

				if(filelist.used){
					strcpy(mypars[s_id].fldfile, filelist.fld_files[i_file].c_str());
					strcpy(mypars[s_id].ligandfile, filelist.ligand_files[i_file].c_str());
				}

				// Filling the filename and coeffs fields of mypars according to command line arguments
				if (get_filenames_and_ADcoeffs(&argc, argv, &(mypars[s_id]), filelist.used) != 0)
					{printf("\n\nError in get_filenames_and_ADcoeffs, stopped job."); err = 1;}

				//------------------------------------------------------------
				// Testing command line arguments for cgmaps parameter
				// since we need it at grid creation time
				//------------------------------------------------------------
				mypars[s_id].cgmaps = 0; // default is 0 (use one maps for every CGx or Gx atom types, respectively)
				for (unsigned int i=1; i<argc-1; i+=2)
				{
					// ----------------------------------
					//Argument: Use individual maps for CG-G0 instead of the same one
					if (strcmp("-cgmaps", argv [i]) == 0)
					{
						int tempint;
						sscanf(argv [i+1], "%d", &tempint);
						if (tempint == 0)
							mypars[s_id].cgmaps = 0;
						else
							mypars[s_id].cgmaps = 1;
					}
				}

				//------------------------------------------------------------
				// Processing receptor and ligand files
				//------------------------------------------------------------

				// Filling mygrid[s_id] according to the gpf file
				if (get_gridinfo(mypars[s_id].fldfile, &(mygrid[s_id])) != 0)
					{printf("\n\nError in get_gridinfo, stopped job."); err = 1;}

				// Filling the atom types filed of myligand according to the grid types
				if (init_liganddata(mypars[s_id].ligandfile, &(myligand_init[s_id]), &(mygrid[s_id]), mypars[s_id].cgmaps) != 0)
					{printf("\n\nError in init_liganddata, stopped job."); err = 1;}

				// Filling myligand according to the pdbqt file
				if (get_liganddata(mypars[s_id].ligandfile, &(myligand_init[s_id]), mypars[s_id].coeffs.AD4_coeff_vdW, mypars[s_id].coeffs.AD4_coeff_hb) != 0)
					{printf("\n\nError in get_liganddata, stopped job."); err = 1;}

				// Resize grid
				Kokkos::resize(floatgrids[s_id], 4*(mygrid[s_id].num_of_atypes+2)*mygrid[s_id].size_xyz[0]*mygrid[s_id].size_xyz[1]*mygrid[s_id].size_xyz[2]);

				//Reading the grid files and storing values in the memory region pointed by floatgrids
				if (get_gridvalues_f(&(mygrid[s_id]), floatgrids[s_id].data(), mypars[s_id].cgmaps) != 0)
					{printf("\n\nError in get_gridvalues_f, stopped job."); err = 1;}

				//------------------------------------------------------------
				// Capturing algorithm parameters (command line args)
				//------------------------------------------------------------
				get_commandpars(&argc, argv, &(mygrid[s_id].spacing), &(mypars[s_id]));

				if (filelist.resnames.size()>0){ // Overwrite resname with specified filename if specified in file list
					strcpy(mypars[s_id].resname, filelist.resnames[i_file].c_str());
				} else if (filelist.used) { // otherwise add the index to existing name distinguish the files if multiple
					std::string if_str = std::to_string(i_file);
					strcat(mypars[s_id].resname, if_str.c_str());
				}

				Gridinfo mydummygrid;
				// if -lxrayfile provided, then read xray ligand data
				if (mypars[s_id].given_xrayligandfile == true) {
					if (init_liganddata(mypars[s_id].xrayligandfile, &(myxrayligand[s_id]), &mydummygrid, mypars[s_id].cgmaps) != 0)
						{printf("\n\nError in init_liganddata, stopped job."); err = 1;}

					if (get_liganddata(mypars[s_id].xrayligandfile, &(myxrayligand[s_id]), mypars[s_id].coeffs.AD4_coeff_vdW, mypars[s_id].coeffs.AD4_coeff_hb) != 0)
						{printf("\n\nError in get_liganddata, stopped job."); err = 1;}
				}

				//------------------------------------------------------------
				// Calculating energies of reference ligand if required
				//------------------------------------------------------------
				if (mypars[s_id].reflig_en_reqired == 1) {
					print_ref_lig_energies_f(myligand_init[s_id],
								 mypars[s_id].smooth,
								 mygrid[s_id],
								 floatgrids[s_id].data(),
								 mypars[s_id].coeffs.scaled_AD4_coeff_elec,
								 mypars[s_id].coeffs.AD4_coeff_desolv,
								 mypars[s_id].qasp);
				}
			}
			// Do the execution on thread 1, except on the first iteration since nothing is loaded yet
			if ((thread_id==execution_thread) && i_file>0) {
				//------------------------------------------------------------
				// Starting Docking
				//------------------------------------------------------------

				if (docking_with_gpu(&(mygrid[r_id]), floatgrids[r_id], &(mypars[r_id]), &(myligand_init[r_id]), &(myxrayligand[r_id]), &argc, argv) != 0)
					{printf("\n\nError in docking_with_gpu, stopped job."); err = 1;}

			}
#ifndef _WIN32
			if (thread_id==setup_thread && overlap) setup_time = seconds_since(loop_time_start);
			if (thread_id==execution_thread && overlap) exec_time = seconds_since(loop_time_start);
#endif
		} // End of openmp parallel region, implicit thread barrier
		if (err==1) return 1; // Couldnt return immediately while in parallel region

#ifndef _WIN32
		// Time measurement of this loop
		double loop_time = seconds_since(loop_time_start);
		printf("\nLoop run time %.3f sec \n", loop_time);
		// Determine overlap savings (no overlap at beginning and end of pipeline)
		if (overlap && i_file>0 && i_file<n_files){
			double savings = overlap ? (setup_time + exec_time - loop_time) : 0;
			total_savings += savings;
			printf("Savings from overlap: %.3f sec \n", savings);
		}

		if (i_file>0){
			// Append time information to .dlg file
			char report_file_name[256];
			strcpy(report_file_name, mypars[r_id].resname);
			strcat(report_file_name, ".dlg");
			FILE* fp = fopen(report_file_name, "a");
			fprintf(fp, "\n\n\nRun time %.3f sec\n", loop_time);
			fclose(fp);
		}
#endif
		if (err==1) return 1;
	} // end of i_file loop

#ifndef _WIN32
	// Total time measurement
	printf("\nRun time of entire job set (%d files): %.3f sec", n_files, seconds_since(time_start));
	if (overlap) printf("\nTotal savings from overlap: %.3f sec \n\n", total_savings); 
#endif

	} // End kokkos scope so that views deallocate before reaching Kokkos:finalize()
	Kokkos::finalize();

	return 0;
}
