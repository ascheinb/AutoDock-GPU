#ifndef CALCENERGRAD_HPP
#define CALCENERGRAD_HPP

/*
template<class Device>
KOKKOS_INLINE_FUNCTION void kokkos_get_atom_pos(const int atom_id, const Conform<Device>& conform, Kokkos::View<float4struct[MAX_NUM_OF_ATOMS]> calc_coords);

template<class Device>
KOKKOS_INLINE_FUNCTION void kokkos_rotate_atoms(const int rotation_counter, const Conform<Device>& conform, const RotList<Device>& rotlist, const int run_id, const float* genotype, const float4struct& genrot_movingvec, const float4struct& genrot_unitvec, Kokkos::View<float4struct[MAX_NUM_OF_ATOMS]> calc_coords);

template<class Device>
KOKKOS_INLINE_FUNCTION float kokkos_calc_intermolecular_energy(const int atom_id, const DockingParams<Device>& dock_params, const InterIntra<Device>& interintra, const Kokkos::View<float4struct[MAX_NUM_OF_ATOMS]> calc_coords);

template<class Device>
KOKKOS_INLINE_FUNCTION float kokkos_calc_intramolecular_energy(const int contributor_counter, const DockingParams<Device>& dock_params, const IntraContrib<Device>& intracontrib, const InterIntra<Device>& interintra, const Intra<Device>& intra, Kokkos::View<float4struct[MAX_NUM_OF_ATOMS]> calc_coords);
*/
template<class Device>
KOKKOS_INLINE_FUNCTION void kokkos_calc_energrad(const member_type& team_member, const DockingParams<Device>& docking_params,const float *genotype,const Conform<Device>& conform, const RotList<Device>& rotlist, const IntraContrib<Device>& intracontrib, const InterIntra<Device>& interintra, const Intra<Device>& intra, const Grads<Device>& grads, const AxisCorrection<Device>& axis_correction, float& energy, float* gradient);

#include "calcenergrad.tpp"

#endif