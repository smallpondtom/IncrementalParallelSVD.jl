"""
    check_orthogonality(M::Matrix{Float64}, comm_size::Int)

Check orthogonality of distributed matrix M by computing dot product of first and last columns.
This requires global communication to sum the local contributions.
"""
function check_orthogonality(M::Matrix{Float64}, comm_size::Int = 1)
    if size(M, 2) <= 1
        return 0.0
    end
    
    # Compute local dot product between first and last columns
    local_result = dot(M[:, 1], M[:, end])
    
    # Reduce across MPI processes if needed
    global_result = local_result
    if MPI.Initialized() && comm_size > 1
        global_result = MPI.Allreduce(local_result, MPI.SUM, MPI.COMM_WORLD)
    end
    
    return global_result
end