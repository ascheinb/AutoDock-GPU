#ifndef COMMON_TYPEDEFS_HPP
#define COMMON_TYPEDEFS_HPP

#include "float4struct.hpp"

// View type on scratch memory that is used in energy and gradient calculations
// Coordinates of all atoms
typedef Kokkos::View<float4struct*,ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> Coordinates;

// Gradient (inter/intra, xyz, num atoms)
typedef Kokkos::View<float**,ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> AtomGradients;

// Genotype
typedef Kokkos::View<float*,ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> Genotype;

// Identical to Genotype, but for auxiliary arrays (e.g. gradient) that arent technically genotypes themselves. To avoid confusion, shouldnt be labeled as a genotype
typedef Kokkos::View<float*,ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> GenotypeAux;

// Array of length team_size for use in perform_elitist_selection
typedef Kokkos::View<float[NUM_OF_THREADS_PER_BLOCK],ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> TeamFloat;
typedef Kokkos::View<int[NUM_OF_THREADS_PER_BLOCK],ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> TeamInt;

// Arrays of different fixed sizes (maybe unnecessary but fixed probably performs better so use it if length is known at compile time)
typedef Kokkos::View<bool[1],ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> OneBool;
typedef Kokkos::View<int[1],ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> OneInt;
typedef Kokkos::View<int[2],ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> TwoInt;
typedef Kokkos::View<int[4],ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> FourInt;
typedef Kokkos::View<float[1],ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> OneFloat;
typedef Kokkos::View<float[4],ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> FourFloat;
typedef Kokkos::View<float[10],ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> TenFloat;

#endif
