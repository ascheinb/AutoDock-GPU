#ifndef GENOTYPE_FUNCS_HPP
#define GENOTYPE_FUNCS_HPP

#include "common_typedefs.hpp"

#define TEAM_PARALLEL_FOR(team_member, len, idx) for (int idx = team_member.team_rank(); idx < len; idx += team_member.team_size())


// Perhaps these could be replaced with kokkos deep_copies, however it may require
// something sophisticated that isnt worth it unless the speedup is large

// global to local copy
template<class Device>
KOKKOS_INLINE_FUNCTION void copy_genotype(const member_type& team_member, const int genotype_length, Genotype genotype, const Generation<Device>& generation, int which_pop)
{
	int offset = GENOTYPE_LENGTH_IN_GLOBMEM*which_pop;
	TEAM_PARALLEL_FOR(team_member, genotype_length, idx)
        	genotype[idx] = generation.conformations(offset + idx);
}

// local to global copy
template<class Device>
KOKKOS_INLINE_FUNCTION void copy_genotype(const member_type& team_member, const int genotype_length, const Generation<Device>& generation, int which_pop, Genotype genotype)
{
        int offset = GENOTYPE_LENGTH_IN_GLOBMEM*which_pop;
        TEAM_PARALLEL_FOR(team_member, genotype_length, idx)
                generation.conformations(offset + idx) = genotype[idx];
}

// local to local copy - note, not a template because Device isnt present.
KOKKOS_INLINE_FUNCTION void copy_genotype(const member_type& team_member, const int genotype_length, Genotype genotype_copy, Genotype genotype)
{
        TEAM_PARALLEL_FOR(team_member, genotype_length, idx)
                genotype_copy(idx) = genotype[idx];
}

// global to global copy
template<class Device>
KOKKOS_INLINE_FUNCTION void copy_genotype(const member_type& team_member, const int genotype_length, const Generation<Device>& generation_copy, int which_pop_copy, const Generation<Device>& generation, int which_pop)
{
        int offset = GENOTYPE_LENGTH_IN_GLOBMEM*which_pop;
	int offset_copy = GENOTYPE_LENGTH_IN_GLOBMEM*which_pop_copy;
        TEAM_PARALLEL_FOR(team_member, genotype_length, idx)
                generation_copy.conformations(offset_copy + idx) = generation.conformations(offset + idx);
}

#endif
