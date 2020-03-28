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
#include <omp.h>
#include <Kokkos_Core.hpp>
#include "kokkos_settings.hpp"

#include "processgrid.h"
//include "processresult.h"
#include "processligand.h"
#include "getparameters.h"
#include "performdocking.h"

#ifndef _WIN32
// ------------------------
// Time measurement
#include <sys/time.h>
// ------------------------
#endif

int main(int argc, char* argv[])
{
  Kokkos::initialize();

  int n_files = 4; // For now, just a loop index
  int err = 0;
#ifndef _WIN32
  // Start full timer
  timeval time_start,time_end, file_time_start,file_time_end;
  gettimeofday(&time_start,NULL);
  double num_sec, num_usec, elapsed_sec;
#endif

  // Objects that are arguments of docking_with_gpu
  // These must each have 2
  Dockpars   mypars[2];
  Liganddata myligand_init[2];
  Gridinfo   mygrid[2];
  Liganddata myxrayligand[2];
  Kokkos::View<float*,HostType> floatgrids0("floatgrids0", 0);
  Kokkos::View<float*,HostType> floatgrids1("floatgrids1", 0);

  for (int i_test=0;i_test<(n_files+1);i_test++){ // one extra iteration since its a pipeline
    int s_id = i_test % 2;    // Alternate which set is undergoing setup (s_id)
    int r_id = (i_test+1) %2; // and which is being used in the run (r_id)

#ifndef _WIN32
    // Time measurement: start of loop
    gettimeofday(&file_time_start,NULL);
#endif
    // Branch into two threads
    // Thread 0 reads files and prepares the inputs to docking_with_gpu
    // Thread 1 runs docking_with_gpu
#ifdef USE_GPU
    #pragma omp parallel
    {
      int thread_id = omp_get_thread_num();
      // Thread 0 does the setup, unless its the last run (so nothing left to load)
      if (thread_id==0 && i_test<n_files) {
#else
      {
      if (i_test<n_files) {
#endif
	//------------------------------------------------------------
	// Capturing names of grid parameter file and ligand pdbqt file
	//------------------------------------------------------------

	// Filling the filename and coeffs fields of mypars according to command line arguments
	if (get_filenames_and_ADcoeffs(&argc, argv, &(mypars[s_id])) != 0)
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
		// ----------------------------------
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

	// Resize grid and set pointer to floatgrids data (maybe should revert this to an array instead of view)
	float* floatgrids;
	if (s_id==0){
		Kokkos::resize(floatgrids0, 4*(mygrid[s_id].num_of_atypes+2)*mygrid[s_id].size_xyz[0]*mygrid[s_id].size_xyz[1]*mygrid[s_id].size_xyz[2]);
		floatgrids = floatgrids0.data();
	} else {
		Kokkos::resize(floatgrids1, 4*(mygrid[s_id].num_of_atypes+2)*mygrid[s_id].size_xyz[0]*mygrid[s_id].size_xyz[1]*mygrid[s_id].size_xyz[2]);
		floatgrids = floatgrids1.data();
	}

	//Reading the grid files and storing values in the memory region pointed by floatgrids
	if (get_gridvalues_f(&(mygrid[s_id]), floatgrids, mypars[s_id].cgmaps) != 0)
		{printf("\n\nError in get_gridvalues_f, stopped job."); err = 1;}

	//------------------------------------------------------------
	// Capturing algorithm parameters (command line args)
	//------------------------------------------------------------
	get_commandpars(&argc, argv, &(mygrid[s_id].spacing), &(mypars[s_id]));

	// Temporary test: add loop# to resname - ALS
	char it_char[1];
	if (i_test==0) it_char[0]='0';
        if (i_test==1) it_char[0]='1';
        if (i_test==2) it_char[0]='2';
	if (i_test==3) it_char[0]='3';
	strcat(mypars[s_id].resname, it_char);

	Gridinfo   mydummygrid;
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
					 floatgrids,
					 mypars[s_id].coeffs.scaled_AD4_coeff_elec,
					 mypars[s_id].coeffs.AD4_coeff_desolv,
					 mypars[s_id].qasp);
	}

      }
#ifdef USE_GPU
      // Do the execution on thread 1, except on the first iteration since nothing is loaded yet
      if (thread_id==1 && i_test>0) {
#else
      if (i_test>0) {
#endif
	//------------------------------------------------------------
	// Starting Docking
	//------------------------------------------------------------

	printf("\nAutoDock-GPU version: %s\n", VERSION);

	if (r_id==0){
		if (docking_with_gpu(&(mygrid[r_id]), floatgrids0, &(mypars[r_id]), &(myligand_init[r_id]), &(myxrayligand[r_id]), &argc, argv) != 0)
			{printf("\n\nError in docking_with_gpu, stopped job."); err = 1;}
	} else {
                if (docking_with_gpu(&(mygrid[r_id]), floatgrids1, &(mypars[r_id]), &(myligand_init[r_id]), &(myxrayligand[r_id]), &argc, argv) != 0)
                        {printf("\n\nError in docking_with_gpu, stopped job."); err = 1;}
	}

      }
    } // End of openmp parallel region

#ifndef _WIN32
    // ------------------------
    // Time measurement of this loop
    gettimeofday(&file_time_end,NULL);
    num_sec     = file_time_end.tv_sec  - file_time_start.tv_sec;
    num_usec    = file_time_end.tv_usec - file_time_start.tv_usec;
    elapsed_sec = num_sec + (num_usec/1000000);
    printf("Loop run time %.3f sec \n\n", elapsed_sec);

    if (i_test>0){
      // Append time information to .dlg file
      char report_file_name[256];
      strcpy(report_file_name, mypars[r_id].resname);
      strcat(report_file_name, ".dlg");
      FILE* fp;
      fp = fopen(report_file_name, "a");
      fprintf(fp, "\n\n\nFile run time %.3f sec\n", elapsed_sec);
      fclose(fp);
    }
#endif
        
    if (err==1) return 1;
  }

#ifndef _WIN32
  // Total time measurement
  gettimeofday(&time_end,NULL);
  num_sec     = time_end.tv_sec  - time_start.tv_sec;
  num_usec    = time_end.tv_usec - time_start.tv_usec;
  elapsed_sec = num_sec + (num_usec/1000000);
  printf("\nRun time of all %d files together: %.3f sec \n\n", n_files, elapsed_sec);

  //// ------------------------
#endif

  Kokkos::finalize();

  return 0;
}
