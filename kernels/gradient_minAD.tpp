#include "calcenergrad.hpp"
#include "ada_functions.hpp"
#include "random.hpp"

// TODO - templatize ExSpace - ALS
template<class Device>
void gradient_minAD(Generation<Device>& next, Dockpars* mypars,DockingParams<Device>& docking_params,Constants<Device>& consts)
{
	Kokkos::parallel_for (Kokkos::RangePolicy<ExSpace,Kokkos::LaunchBounds<NUM_OF_THREADS_PER_BLOCK>> (0, docking_params.num_of_lsentities * mypars->num_of_runs ),
                        KOKKOS_LAMBDA (const int idx)
        {
		// Determine gpop_idx
		int run_id = idx / docking_params.num_of_lsentities;
		int entity_id = idx - run_id * docking_params.num_of_lsentities; // modulus in different form

		// Since entity 0 is the best one due to elitism,
		// it should be subjected to random selection
		if (entity_id == 0) {
			// If entity 0 is not selected according to LS-rate, choosing another entity
			if (100.0f*rand_float(idx, docking_params) > docking_params.lsearch_rate) {
				entity_id = docking_params.num_of_lsentities; // AT - Should this be (uint)(dockpars_pop_size * gpu_randf(dockpars_prng_states))?
			}
		}

		int gpop_idx = run_id*docking_params.pop_size+entity_id; // global population index

		// Copy genotype to local shared memory
                float genotype[ACTUAL_GENOTYPE_LENGTH];
		copy_genotype(docking_params.num_of_genes, genotype, next, gpop_idx);

		// Initializing best genotype and energy
		float energy; // Dont need to init this since it's overwritten
		float best_energy = INFINITY;
		float best_genotype[ACTUAL_GENOTYPE_LENGTH];
                copy_genotype(docking_params.num_of_genes, best_genotype, genotype);

		// Initializing variable arrays for gradient descent
		float square_gradient[ACTUAL_GENOTYPE_LENGTH];
		float square_delta[ACTUAL_GENOTYPE_LENGTH];
		for(int i = 0; i < ACTUAL_GENOTYPE_LENGTH; i++ ) {
                        square_gradient[i]=0; // Probably unnecessary since kokkos views are automatically initialized to 0 (not sure if that's the case in scratch though)
			square_delta[i]=0;
                }


		// Initialize iteration controls
		bool stay_in_loop=true;
		bool energy_improved=false;
		unsigned int iteration_cnt = 0;

#ifdef AD_RHO_CRITERION
		float rho = 1.0f;
		int   cons_succ = 0;
		int   cons_fail = 0;
#endif

		// Perform adadelta iterations on gradient
		float gradient[ACTUAL_GENOTYPE_LENGTH];
		// The termination criteria is based on
		// a maximum number of iterations, and
		// the minimum step size allowed for single-floating point numbers
		// (IEEE-754 single float has a precision of about 6 decimal digits)
		do {
			// Calculating energy & gradient
			calc_energrad(run_id, docking_params, genotype, consts,
					     energy, gradient);

			if (energy < best_energy) energy_improved=true;

			// we need to be careful not to change best_energy until we had a chance to update the whole array
			if (energy_improved){
				copy_genotype(docking_params.num_of_genes, best_genotype, genotype);
				best_energy = energy;
			}

			// Update genotype based on gradient
			genotype_gradient_descent(docking_params, gradient, square_gradient, square_delta, genotype);

			// Iteration controls
#ifdef AD_RHO_CRITERION
			if (energy_improved) {
				cons_succ++;
				cons_fail = 0;
			} else {
				cons_succ = 0;
				cons_fail++;
			}

			if (cons_succ >= 4) {
				rho *= LS_EXP_FACTOR;
				cons_succ = 0;
			} else if (cons_fail >= 4) {
				rho *= LS_CONT_FACTOR;
				cons_fail = 0;
			}
#endif
#if defined (PRINT_ADADELTA_MINIMIZER_ENERGY_EVOLUTION)
			printf("%-15s %-3u ", "# ADADELTA iteration: ", iteration_cnt);
			printf("%20s %10.6f\n", "new.energy: ", energy);
#endif
			// Updating number of ADADELTA iterations (energy evaluations)
			iteration_cnt = iteration_cnt + 1;
			energy_improved=false; // reset to zero for next loop iteration

#ifdef AD_RHO_CRITERION
			if ((iteration_cnt >= docking_params.max_num_of_iters) || (rho <= 0.01))
#else
			if (iteration_cnt >= docking_params.max_num_of_iters)
#endif
				stay_in_loop=false;

		} while (stay_in_loop);
		// Descent complete
		// -----------------------------------------------------------------------------

		// Modulo torsion angles
		for (int gene_counter = 3; gene_counter < docking_params.num_of_genes; gene_counter++ ) {
                        while (best_genotype[gene_counter] >= 360.0f) { best_genotype[gene_counter] -= 360.0f; }
                        while (best_genotype[gene_counter] < 0.0f   ) { best_genotype[gene_counter] += 360.0f; }
		}

                // Copy to global views
                next.energies(gpop_idx) = best_energy;
                docking_params.evals_of_new_entities(gpop_idx) += iteration_cnt;

		copy_genotype(docking_params.num_of_genes, next, gpop_idx, best_genotype);
        });
}
