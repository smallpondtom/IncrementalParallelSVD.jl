"""
Random data example.
"""

##
using MPI
using IncrementalParallelSVD
MPI.Init()

# Create an incremental SVD instance
dim = 10000  # Local dimension
isvd = create_incremental_svd_brand(
    dim = dim,
    linearity_tol = 1e-12,
    max_basis_dimension = 50,
    update_right_SV = true
)

## Add samples
if dim < 10^6
    n_samples = 20  # Number of samples
    X = randn(dim, n_samples)
    for i in 1:n_samples
        x = X[:, i]
        success = take_sample!(isvd, x)
        if !success
            println("Failed to add sample $i")
        end
    end
else
    n_samples = 2000  # Number of samples
    for i in 1:n_samples
        x = randn(dim)
        success = take_sample!(isvd, x)
        if !success
            println("Failed to add sample $i")
        end
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
if dim < 10^6
    using LinearAlgebra: svd, norm, Diagonal
    U, S, V = svd(X)
    U = U[:, 1:size(spatial_basis, 2)]
    S = S[1:size(spatial_basis, 2)]  

    ## Check errors
    println("Error in singular values: $(norm(singular_values - S))")
    println(
        "Error in spatial basis: 
        $(norm(spatial_basis * spatial_basis' - U * U') / sqrt(2))
    ")
end
