#ifndef CALCENERGY_HPP
#define CALCENERGY_HPP

#include "float4struct.hpp"
#include "common_typedefs.hpp"

template<class Device>
KOKKOS_INLINE_FUNCTION void get_atom_pos(const int atom_id, const Conform<Device>& conform, Coordinates calc_coords);

template<class Device>
KOKKOS_INLINE_FUNCTION void rotate_atoms(const int rotation_counter, const Conform<Device>& conform, const RotList<Device>& rotlist, const int run_id, Genotype genotype, const float4struct& genrot_movingvec, const float4struct& genrot_unitvec, Coordinates calc_coords);

template<class Device>
KOKKOS_INLINE_FUNCTION float calc_intermolecular_energy(const int atom_id, const DockingParams<Device>& dock_params, const InterIntra<Device>& interintra, const Kokkos::View<float4struct[MAX_NUM_OF_ATOMS]> calc_coords);

template<class Device>
KOKKOS_INLINE_FUNCTION float calc_intramolecular_energy(const int contributor_counter, const DockingParams<Device>& dock_params, const IntraContrib<Device>& intracontrib, const InterIntra<Device>& interintra, const Intra<Device>& intra, Kokkos::View<float4struct[MAX_NUM_OF_ATOMS]> calc_coords);

template<class Device>
KOKKOS_INLINE_FUNCTION float calc_energy(const member_type& team_member, const DockingParams<Device>& docking_params, const Constants<Device>& consts, Coordinates calc_coords, Genotype genotype, int run_id);

// Non-hierarchical versions
template<class Device>
KOKKOS_INLINE_FUNCTION void get_atom_pos(const int idx, const int atom_id, const Conform<Device>& conform, const DockingParams<Device>& docking_params);

template<class Device>
KOKKOS_INLINE_FUNCTION void rotate_atoms(const int idx, const int rotation_counter, const Conform<Device>& conform, const RotList<Device>& rotlist, const int run_id, float* genotype, const float4struct& genrot_movingvec, const float4struct& genrot_unitvec, const DockingParams<Device>& docking_params);

#include "calcenergy.tpp"

#endif
