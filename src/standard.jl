"""
IncrementalSVDStandard

Standard incremental SVD algorithm implementation.
"""
mutable struct IncrementalSVDStandard <: AbstractIncrementalSVD
    base::IncrementalSVD
    
    function IncrementalSVDStandard(options::SVDOptions, basis_file_name::String, dim::Int)
        base = IncrementalSVD(options, basis_file_name, dim)
        svd_obj = new(base)
        
        if options.restore_state
            restore_standard_state!(svd_obj)
        end
        
        return svd_obj
    end
end

function take_sample!(svd_obj::IncrementalSVDStandard, u::Vector{Float64}; add_without_increase::Bool=false)
    @assert length(u) == svd_obj.base.dim "Input vector dimension mismatch"
    
    # Check if input is non-zero
    if norm(u) ≈ 0.0
        return false
    end
    
    # Build initial SVD or add incremental sample
    if svd_obj.base.is_first_sample
        build_initial_svd!(svd_obj, u)
        svd_obj.base.is_first_sample = false
    else
        result = build_incremental_svd!(svd_obj, u, add_without_increase)
        if !result
            return false
        end
    end
    
    # Debug output
    if svd_obj.base.debug_algorithm
        debug_output(svd_obj.base)
    end
    
    return true
end

get_singular_values(svd_obj::IncrementalSVDStandard) = get_singular_values(svd_obj.base)
get_spatial_basis(svd_obj::IncrementalSVDStandard) = get_spatial_basis(svd_obj.base)
get_temporal_basis(svd_obj::IncrementalSVDStandard) = get_temporal_basis(svd_obj.base)

"""
    restore_standard_state!(svd_obj::IncrementalSVDStandard)

Restore Standard-specific state (just calls base restore and computes basis).
"""
function restore_standard_state!(svd_obj::IncrementalSVDStandard)
    if svd_obj.base.U !== nothing
        compute_basis!(svd_obj)
        return true
    end
    return false
end

"""
    build_initial_svd!(svd_obj::IncrementalSVDStandard, u::Vector{Float64})

Build initial SVD for Standard algorithm.
"""
function build_initial_svd!(svd_obj::IncrementalSVDStandard, u::Vector{Float64})
    norm_u = norm(u)
    
    # Initialize S
    svd_obj.base.S = [norm_u]
    
    # Initialize U
    svd_obj.base.U = reshape(u / norm_u, :, 1)
    
    # Initialize W if needed
    if svd_obj.base.update_right_SV
        svd_obj.base.W = reshape([1.0], 1, 1)
    end
    
    # Compute basis immediately
    compute_basis!(svd_obj)
    
    svd_obj.base.num_samples = 1
    svd_obj.base.num_rows_of_W = 1
end

"""
    compute_basis!(svd_obj::IncrementalSVDStandard)

Compute basis for Standard algorithm (direct copy of U and W).
"""
function compute_basis!(svd_obj::IncrementalSVDStandard)
    # Direct copy for standard algorithm
    svd_obj.base.basis = copy(svd_obj.base.U)
    
    if svd_obj.base.update_right_SV
        svd_obj.base.basis_right = copy(svd_obj.base.W)
    end
end

"""
    add_linearly_dependent_sample!(svd_obj::IncrementalSVDStandard, A::Matrix{Float64}, W::Matrix{Float64}, sigma::Matrix{Float64})

Add linearly dependent sample for Standard algorithm.
"""
function add_linearly_dependent_sample!(svd_obj::IncrementalSVDStandard, A::Matrix{Float64}, W::Matrix{Float64}, sigma::Matrix{Float64})
    n = svd_obj.base.num_samples
    
    # Extract submatrices
    A_mod = A[1:n, 1:n]
    
    # Update singular values
    for i in 1:n
        svd_obj.base.S[i] = sigma[i, i]
    end
    
    # Update U
    svd_obj.base.U = svd_obj.base.U * A_mod
    
    # Update W if needed
    if svd_obj.base.update_right_SV
        new_W = zeros(svd_obj.base.num_rows_of_W + 1, n)
        
        # Update existing rows
        for i in 1:svd_obj.base.num_rows_of_W
            for j in 1:n
                new_W[i, j] = sum(svd_obj.base.W[i, k] * W[k, j] for k in 1:n)
            end
        end
        
        # Add new row
        for j in 1:n
            new_W[svd_obj.base.num_rows_of_W + 1, j] = W[n + 1, j]
        end
        
        svd_obj.base.W = new_W
        svd_obj.base.num_rows_of_W += 1
    end
    
    # Reorthogonalize if necessary
    max_dim = max(svd_obj.base.num_samples, svd_obj.base.total_dim)
    tol = eps(Float64) * max_dim
    
    if abs(check_orthogonality(svd_obj.base.U)) > tol
        F = qr(svd_obj.base.U)
        svd_obj.base.U = Matrix(F.Q)
    end
end

"""
    add_new_sample!(svd_obj::IncrementalSVDStandard, j::Vector{Float64}, A::Matrix{Float64}, W::Matrix{Float64}, sigma::Matrix{Float64})

Add new linearly independent sample for Standard algorithm.
"""
function add_new_sample!(svd_obj::IncrementalSVDStandard, j::Vector{Float64}, A::Matrix{Float64}, W::Matrix{Float64}, sigma::Matrix{Float64})
    n = svd_obj.base.num_samples
    
    # Create temporary matrix with j as new column
    tmp = hcat(svd_obj.base.U, j)
    
    # Multiply by A to get new U
    svd_obj.base.U = tmp * A
    
    # Update W if needed
    if svd_obj.base.update_right_SV
        new_W = zeros(svd_obj.base.num_rows_of_W + 1, n + 1)
        
        # Update existing part
        for i in 1:svd_obj.base.num_rows_of_W
            for j_idx in 1:(n + 1)
                new_W[i, j_idx] = sum(svd_obj.base.W[i, k] * W[k, j_idx] for k in 1:n)
            end
        end
        
        # Add new row
        for j_idx in 1:(n + 1)
            new_W[svd_obj.base.num_rows_of_W + 1, j_idx] = W[n + 1, j_idx]
        end
        
        svd_obj.base.W = new_W
    end
    
    # Update singular values
    num_dim = min(size(sigma, 1), size(sigma, 2))
    svd_obj.base.S = [sigma[i, i] for i in 1:num_dim]
    
    # Increment counters
    svd_obj.base.num_samples += 1
    svd_obj.base.num_rows_of_W += 1
    
    # Reorthogonalize if necessary
    max_dim = max(svd_obj.base.num_samples, svd_obj.base.total_dim)
    tol = eps(Float64) * max_dim
    
    if abs(check_orthogonality(svd_obj.base.U)) > tol
        F = qr(svd_obj.base.U)
        svd_obj.base.U = Matrix(F.Q)
    end
    
    if svd_obj.base.update_right_SV && abs(check_orthogonality(svd_obj.base.W)) > eps(Float64) * svd_obj.base.num_samples
        F = qr(svd_obj.base.W)
        svd_obj.base.W = Matrix(F.Q)
    end
end

"""
    build_incremental_svd!(svd_obj::IncrementalSVDStandard, u::Vector{Float64}, add_without_increase::Bool)

Standard incremental SVD implementation.
"""
function build_incremental_svd!(svd_obj::IncrementalSVDStandard, u::Vector{Float64}, add_without_increase::Bool)
    # Compute projection: l = U' * u
    l = svd_obj.base.U' * u
    
    # Compute basis projection: basis_l = U * l
    basis_l = svd_obj.base.U * l
    
    # Compute residual more accurately to avoid catastrophic cancellation
    e_proj = u - basis_l
    k = norm(e_proj)
    
    if k <= 0
        if svd_obj.base.comm_rank == 0
            println("Linearly dependent sample detected!")
        end
        k = 0.0
    end
    
    # Determine if sample is linearly dependent
    linearly_dependent = false
    
    if k < svd_obj.base.linearity_tol
        if svd_obj.base.comm_rank == 0
            println("Linearly dependent sample! k = $k")
            println("linearity_tol = $(svd_obj.base.linearity_tol)")
        end
        k = 0.0
        linearly_dependent = true
    elseif svd_obj.base.num_samples >= svd_obj.base.max_basis_dimension || add_without_increase
        k = 0.0
        linearly_dependent = true
    elseif svd_obj.base.num_samples >= svd_obj.base.total_dim
        linearly_dependent = true
    end
    
    # Construct Q matrix for SVD
    Q = construct_Q(svd_obj.base, l, k)
    
    # Perform SVD of Q
    try
        F = svd(Q)
        U_q, S_q, V_q = F.U, F.S, F.Vt'
        
        # Convert to matrices for consistency
        A = Matrix(U_q)
        sigma = diagm(S_q)
        W = Matrix(V_q)
        
        # Add sample based on linear dependence
        if linearly_dependent && !svd_obj.base.skip_linearly_dependent
            if svd_obj.base.comm_rank == 0
                println("Adding linearly dependent sample!")
            end
            add_linearly_dependent_sample!(svd_obj, A, W, sigma)
        elseif !linearly_dependent
            # Compute normalized residual direction
            j = e_proj / k
            add_new_sample!(svd_obj, j, A, W, sigma)
        end
        
        # Compute basis vectors
        compute_basis!(svd_obj)
        
        return true
    catch e
        @warn "SVD computation failed: $e"
        return false
    end
end