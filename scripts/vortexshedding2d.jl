"""
2D vortex shedding example. Original data from 
https://github.com/ionutfarcas/tutorial_data_driven_modeling/blob/main/TimeDomain/VortexShedding2D/velocity_training_snapshots.h5
"""

##
using MPI
using HDF5
using IncrementalParallelSVD
MPI.Init()

## Load the data
# DoF setup
ns = 2
n = 18954
nx = Int(n / ns)
nt = 300  # Number of training snapshots.

# state variable names
state_variables = ["u_x", "u_y"]

# path to the HDF5 file containing the training snapshots.
H5_training_snapshots = joinpath(@__DIR__, "data/velocity_training_snapshots.h5")

# Allocate memory for the full snapshot data.
# the full snapshot data has been saved to disk in HDF5 format
X = zeros(n, nt)
h5open(H5_training_snapshots, "r") do hf
    for j in 1:ns
        X[(j-1)*nx + 1 : j*nx, :] = read(hf[state_variables[j]])'
    end
end

## Create an incremental SVD instance
dim = n  # Local dimension
isvd = create_incremental_svd_brand(
    dim = dim,
    linearity_tol = 1e-12,
    max_basis_dimension = nt,
    update_right_SV = true
)

## Add samples
for i in 1:nt
    x = X[:, i]
    success = take_sample!(isvd, x)
    if !success
        println("Failed to add sample $i")
    end
end

## Get results
spatial_basis = get_spatial_basis(isvd)
temporal_basis = get_temporal_basis(isvd)
singular_values = get_singular_values(isvd)

println("Computed $(length(singular_values)) singular values")
println("Spatial basis size: $(size(spatial_basis))")

## Clean up
finalize!(isvd)
MPI.Finalize()

## Batch SVD
using LinearAlgebra: svd, norm, Diagonal
U, S, V = svd(X)
U = U[:, 1:size(spatial_basis, 2)]
S = S[1:size(spatial_basis, 2)]  

## Check errors
println("Error in singular values: $(norm(singular_values - S))")
println(
    "Error in spatial basis: 
    $(norm(spatial_basis * spatial_basis' - U * U') / norm(U * U')))
")