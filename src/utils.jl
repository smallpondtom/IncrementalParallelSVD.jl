"""
    check_orthogonality(M::Matrix{Float64})

Check orthogonality of matrix M by computing dot product of first and last columns.
"""
function check_orthogonality(M::Matrix{Float64})
    if size(M, 2) <= 1
        return 0.0
    end
    
    # Compute dot product between first and last columns
    result = dot(M[:, 1], M[:, end])
    
    # Reduce across MPI processes if needed
    if MPI.Initialized()
        result = MPI.Allreduce(result, MPI.SUM, MPI.COMM_WORLD)
    end
    
    return result
end
