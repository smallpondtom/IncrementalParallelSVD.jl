"""
IncrementalSVDBrand

Brand's fast update incremental SVD algorithm implementation.
"""
mutable struct IncrementalSVDBrand <: AbstractIncrementalSVD
    base::IncrementalSVD
    Up::Union{Matrix{Float64}, Nothing}      # U' matrix (not distributed)
    singular_value_tol::Float64              # Tolerance for removing small singular values
    
    function IncrementalSVDBrand(options::SVDOptions, basis_file_name::String, dim::Int)
        base = IncrementalSVD(options, basis_file_name, dim)
        svd_obj = new(base, nothing, options.singular_value_tol)
        
        # Restore additional state if needed
        if options.restore_state
            restore_brand_state!(svd_obj)
        end
        
        return svd_obj
    end
end

"""
    restore_brand_state!(svd_obj::IncrementalSVDBrand)

Restore Brand-specific state (Up matrix).
"""
function restore_brand_state!(svd_obj::IncrementalSVDBrand)
    if !isfile(svd_obj.base.state_file_name)
        return false
    end
    
    try
        h5open(svd_obj.base.state_file_name, "r") do file
            if haskey(file, "Up")
                svd_obj.Up = read(file, "Up")
            end
        end
        
        if svd_obj.Up !== nothing
            compute_basis!(svd_obj)
        end
        
        return true
    catch e
        @warn "Failed to restore Brand state: $e"
        return false
    end
end

"""
    save_brand_state!(svd_obj::IncrementalSVDBrand)

Save Brand-specific state (Up matrix).
"""
function save_brand_state!(svd_obj::IncrementalSVDBrand)
    if !svd_obj.base.save_state || svd_obj.base.is_first_sample
        return
    end
    
    try
        h5open(svd_obj.base.state_file_name, "r+") do file
            if svd_obj.Up !== nothing
                write(file, "Up", svd_obj.Up)
            end
        end
    catch e
        @warn "Failed to save Brand state: $e"
    end
end

get_singular_values(svd_obj::IncrementalSVDBrand) = get_singular_values(svd_obj.base)

"""
    get_spatial_basis(svd_obj::IncrementalSVDBrand)

Returns spatial basis for Brand algorithm with lazy computation.
"""
function get_spatial_basis(svd_obj::IncrementalSVDBrand)
    update_spatial_basis!(svd_obj)
    return svd_obj.base.basis
end

"""
    get_temporal_basis(svd_obj::IncrementalSVDBrand)

Returns temporal basis for Brand algorithm with lazy computation.
"""
function get_temporal_basis(svd_obj::IncrementalSVDBrand)
    update_temporal_basis!(svd_obj)
    return svd_obj.base.basis_right
end

"""
    add_linearly_dependent_sample!(svd_obj::IncrementalSVDBrand, A::Matrix{Float64}, W::Matrix{Float64}, sigma::Matrix{Float64})

Add linearly dependent sample for Brand algorithm.
"""
function add_linearly_dependent_sample!(svd_obj::IncrementalSVDBrand, A::Matrix{Float64}, W::Matrix{Float64}, sigma::Matrix{Float64})
    n = svd_obj.base.num_samples
    
    # Extract submatrices
    A_mod = A[1:n, 1:n]
    
    # Update singular values
    for i in 1:n
        svd_obj.base.S[i] = sigma[i, i]
    end
    
    # Update Up
    svd_obj.Up = svd_obj.Up * A_mod
    
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
end

"""
    update_spatial_basis!(svd_obj::IncrementalSVDBrand)

Update spatial basis with singular value filtering and proper orthogonality checking.
"""
function update_spatial_basis!(svd_obj::IncrementalSVDBrand)
    svd_obj.base.basis = svd_obj.base.U * svd_obj.Up
    
    # Remove small singular values
    if svd_obj.singular_value_tol > 0.0 && svd_obj.base.num_samples > 1
        if svd_obj.base.S[end] < svd_obj.singular_value_tol
            if svd_obj.base.comm_rank == 0
                println("Removing spatial basis corresponding to small singular value!")
            end
            
            svd_obj.base.basis = svd_obj.base.basis[:, 1:end-1]
        end
    end
    
    # Reorthogonalize if necessary (FIXED: pass comm_size for distributed check)
    if abs(check_orthogonality(svd_obj.base.basis, svd_obj.base.comm_size)) > eps(Float64) * svd_obj.base.num_samples
        F = qr(svd_obj.base.basis)
        svd_obj.base.basis = Matrix(F.Q)
    end
end

"""
    update_temporal_basis!(svd_obj::IncrementalSVDBrand)

Update temporal basis with singular value filtering and proper orthogonality checking.
"""
function update_temporal_basis!(svd_obj::IncrementalSVDBrand)
    if svd_obj.base.W === nothing
        return
    end
    
    svd_obj.base.basis_right = copy(svd_obj.base.W)
    
    # Remove small singular values
    if svd_obj.singular_value_tol > 0.0 && svd_obj.base.num_samples > 1
        if svd_obj.base.S[end] < svd_obj.singular_value_tol
            if svd_obj.base.comm_rank == 0
                println("Removing temporal basis corresponding to small singular value!")
            end
            
            svd_obj.base.basis_right = svd_obj.base.basis_right[:, 1:end-1]
        end
    end
    
    # Reorthogonalize if necessary (FIXED: pass 1 for replicated matrix)
    if abs(check_orthogonality(svd_obj.base.basis_right, 1)) > eps(Float64) * svd_obj.base.num_samples
        F = qr(svd_obj.base.basis_right)
        svd_obj.base.basis_right = Matrix(F.Q)
    end
end

"""
    compute_basis!(svd_obj::IncrementalSVDBrand)

Compute both spatial and temporal bases.
"""
function compute_basis!(svd_obj::IncrementalSVDBrand)
    if svd_obj.base.comm_rank == 0
        println("num_samples = $(svd_obj.base.num_samples)")
        println("num_rows_of_W = $(svd_obj.base.num_rows_of_W)")
        println("singular_value_tol = $(svd_obj.singular_value_tol)")
        println("smallest SV = $(svd_obj.base.S[end])")
        if svd_obj.base.num_samples > 1
            println("next smallest SV = $(svd_obj.base.S[end-1])")
        end
    end
    
    update_spatial_basis!(svd_obj)
    
    if svd_obj.base.update_right_SV
        update_temporal_basis!(svd_obj)
    end
    
    # Remove small singular value from count
    if svd_obj.singular_value_tol > 0.0 && svd_obj.base.num_samples > 1
        if svd_obj.base.S[end] < svd_obj.singular_value_tol
            svd_obj.base.num_samples -= 1
        end
    end
end


# Fixed brand.jl - Updated for proper parallelization

function take_sample!(svd_obj::IncrementalSVDBrand, u_local::Vector{Float64}; add_without_increase::Bool=false)
    @assert length(u_local) == svd_obj.base.dim "Input vector dimension mismatch"
    
    # Check if input is non-zero (global norm)
    local_norm_sq = dot(u_local, u_local)
    global_norm_sq = local_norm_sq
    if MPI.Initialized() && svd_obj.base.comm_size > 1
        global_norm_sq = MPI.Allreduce(local_norm_sq, MPI.SUM, MPI.COMM_WORLD)
    end
    
    if sqrt(global_norm_sq) ≈ 0.0
        return false
    end
    
    # Build initial SVD or add incremental sample
    if svd_obj.base.is_first_sample
        build_initial_svd!(svd_obj, u_local)
        svd_obj.base.is_first_sample = false
    else
        result = build_incremental_svd!(svd_obj, u_local, add_without_increase)
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

"""
    build_initial_svd!(svd_obj::IncrementalSVDBrand, u_local::Vector{Float64})

Build initial SVD for Brand algorithm using distributed vector u_local.
"""
function build_initial_svd!(svd_obj::IncrementalSVDBrand, u_local::Vector{Float64})
    # Compute global norm
    local_norm_sq = dot(u_local, u_local)
    global_norm_sq = local_norm_sq
    if MPI.Initialized() && svd_obj.base.comm_size > 1
        global_norm_sq = MPI.Allreduce(local_norm_sq, MPI.SUM, MPI.COMM_WORLD)
    end
    norm_u = sqrt(global_norm_sq)
    
    # Initialize S
    svd_obj.base.S = [norm_u]
    
    # Initialize Up
    svd_obj.Up = reshape([1.0], 1, 1)
    
    # Initialize U (local portion)
    svd_obj.base.U = reshape(u_local / norm_u, :, 1)
    
    # Initialize W if needed
    if svd_obj.base.update_right_SV
        svd_obj.base.W = reshape([1.0], 1, 1)
    end
    
    svd_obj.base.num_samples = 1
    svd_obj.base.num_rows_of_W = 1
end

"""
    build_incremental_svd!(svd_obj::IncrementalSVDBrand, u_local::Vector{Float64}, add_without_increase::Bool)

Brand-specific incremental SVD implementation for distributed vectors.
"""
function build_incremental_svd!(svd_obj::IncrementalSVDBrand, u_local::Vector{Float64}, add_without_increase::Bool)
    # Compute projection error using Gram-Schmidt with re-orthogonalization
    e_proj_local = copy(u_local)
    
    # First Gram-Schmidt step (requires global communication)
    tmp_local = svd_obj.base.U' * e_proj_local
    tmp = copy(tmp_local)
    if MPI.Initialized() && svd_obj.base.comm_size > 1
        MPI.Allreduce!(tmp, MPI.SUM, MPI.COMM_WORLD)
    end
    e_proj_local -= svd_obj.base.U * tmp
    
    # Re-orthogonalization step (requires global communication)
    tmp_local = svd_obj.base.U' * e_proj_local
    tmp = copy(tmp_local)
    if MPI.Initialized() && svd_obj.base.comm_size > 1
        MPI.Allreduce!(tmp, MPI.SUM, MPI.COMM_WORLD)
    end
    e_proj_local -= svd_obj.base.U * tmp
    
    # Compute global norm
    k_local_sq = dot(e_proj_local, e_proj_local)
    k_global_sq = k_local_sq
    if MPI.Initialized() && svd_obj.base.comm_size > 1
        k_global_sq = MPI.Allreduce(k_local_sq, MPI.SUM, MPI.COMM_WORLD)
    end
    k = sqrt(k_global_sq)
    
    if k <= 0
        if svd_obj.base.comm_rank == 0
            println("Linearly dependent sample!")
        end
        k = 0.0
    end
    
    # Check linear dependence
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
    
    # Construct Q matrix
    U_mult_u_local = svd_obj.base.U' * u_local
    U_mult_u = copy(U_mult_u_local)
    if MPI.Initialized() && svd_obj.base.comm_size > 1
        MPI.Allreduce!(U_mult_u, MPI.SUM, MPI.COMM_WORLD)
    end
    
    l = svd_obj.Up' * U_mult_u
    Q = construct_Q(svd_obj.base, l, k)
    
    # Perform SVD
    try
        F = svd(Q)
        U_q, S_q, V_q = F.U, F.S, F.Vt'
        
        A = Matrix(U_q)
        sigma = diagm(S_q)
        W = Matrix(V_q)
        
        # Add sample
        if linearly_dependent && !svd_obj.base.skip_linearly_dependent
            if svd_obj.base.comm_rank == 0
                println("Adding linearly dependent sample!")
            end
            add_linearly_dependent_sample!(svd_obj, A, W, sigma)
        elseif !linearly_dependent
            j_local = e_proj_local / k
            add_new_sample!(svd_obj, j_local, A, W, sigma)
        end
        
        return true
    catch e
        @warn "SVD computation failed: $e"
        return false
    end
end

"""
    add_new_sample!(svd_obj::IncrementalSVDBrand, j_local::Vector{Float64}, A::Matrix{Float64}, W::Matrix{Float64}, sigma::Matrix{Float64})

Add new linearly independent sample for Brand algorithm (j_local is the local portion).
"""
function add_new_sample!(svd_obj::IncrementalSVDBrand, j_local::Vector{Float64}, A::Matrix{Float64}, W::Matrix{Float64}, sigma::Matrix{Float64})
    n = svd_obj.base.num_samples
    
    # Add j_local as new column to U (local operation)
    new_U = hcat(svd_obj.base.U, j_local)
    svd_obj.base.U = new_U
    
    # Update W if needed (replicated operation)
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
    
    # Update Up (replicated operation)
    new_Up = zeros(n + 1, n + 1)
    
    # Update existing part
    for i in 1:n
        for j_idx in 1:(n + 1)
            new_Up[i, j_idx] = sum(svd_obj.Up[i, k] * A[k, j_idx] for k in 1:n)
        end
    end
    
    # Add new row
    for j_idx in 1:(n + 1)
        new_Up[n + 1, j_idx] = A[n + 1, j_idx]
    end
    
    svd_obj.Up = new_Up
    
    # Update singular values (replicated operation)
    num_dim = min(size(sigma, 1), size(sigma, 2))
    svd_obj.base.S = [sigma[i, i] for i in 1:num_dim]
    
    # Increment counters
    svd_obj.base.num_samples += 1
    svd_obj.base.num_rows_of_W += 1
    
    # Check orthogonality and reorthogonalize if necessary
    max_dim = max(svd_obj.base.num_samples, svd_obj.base.total_dim)
    tol = eps(Float64) * max_dim
    
    # Up is replicated, so check locally
    if abs(check_orthogonality(svd_obj.Up, 1)) > tol
        F = qr(svd_obj.Up)
        svd_obj.Up = Matrix(F.Q)
    end
    
    # U is distributed, so check globally
    if abs(check_orthogonality(svd_obj.base.U, svd_obj.base.comm_size)) > tol
        F = qr(svd_obj.base.U)
        svd_obj.base.U = Matrix(F.Q)
    end
    
    if svd_obj.base.update_right_SV && abs(check_orthogonality(svd_obj.base.W, 1)) > eps(Float64) * svd_obj.base.num_samples
        F = qr(svd_obj.base.W)
        svd_obj.base.W = Matrix(F.Q)
    end
end