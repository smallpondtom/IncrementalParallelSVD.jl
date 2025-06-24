"""
Incremental Singular Value Decomposition (SVD) Implementation in Julia

This module provides parallelized incremental SVD algorithms including:
- IncrementalSVDBrand: Brand's fast update method
- IncrementalSVDFastUpdate: Brand's fast update method (variant)
- IncrementalSVDStandard: Standard incremental SVD algorithm

REFERENCE: Original C++ implementation from libROM project.
https://www.librom.net/
"""
module IncrementalParallelSVD

# Import necessary packages
using LinearAlgebra
using HDF5
using Printf
using MPI

# Abstract base class for incremental SVD algorithms
abstract type AbstractIncrementalSVD end

# Include files
include("options.jl")
include("utils.jl")
include("base.jl")
include("brand.jl")
include("fastupdate.jl")
include("standard.jl")
include("initialize.jl")
include("finalize.jl")

# Export main functions and types
export SVDOptions
export AbstractIncrementalSVD, IncrementalSVD
export IncrementalSVDBrand, IncrementalSVDFastUpdate, IncrementalSVDStandard
export create_incremental_svd_brand, create_incremental_svd_fast_update, create_incremental_svd_standard
export take_sample!, get_spatial_basis, get_temporal_basis, get_singular_values
export finalize!

end # module IncrementalParallelSVD