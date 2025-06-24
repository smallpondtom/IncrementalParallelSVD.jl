# Base incremental SVD type
mutable struct IncrementalSVD <: AbstractIncrementalSVD
    # Core matrices
    U::Union{Matrix{Float64}, Nothing}        # Left singular vectors
    S::Union{Vector{Float64}, Nothing}        # Singular values
    W::Union{Matrix{Float64}, Nothing}        # Right singular vectors
    basis::Union{Matrix{Float64}, Nothing}    # Cached spatial basis
    basis_right::Union{Matrix{Float64}, Nothing}  # Cached temporal basis
    
    # Configuration
    dim::Int                                 # Local dimension
    linearity_tol::Float64                   # Tolerance for linear dependence
    skip_linearly_dependent::Bool            # Whether to skip linearly dependent samples
    max_basis_dimension::Int                 # Maximum basis dimension
    save_state::Bool                         # Whether to save state
    update_right_SV::Bool                    # Whether to update right singular vectors
    debug_algorithm::Bool                    # Debug flag
    
    # MPI information
    comm_size::Int                           # Number of MPI processes
    comm_rank::Int                           # MPI rank
    proc_dims::Vector{Int}                   # Dimensions for each process
    total_dim::Int                           # Total dimension across all processes
    
    # State tracking
    num_samples::Int                        # Number of samples processed
    num_rows_of_W::Int                      # Number of rows in W matrix
    is_first_sample::Bool                   # Flag for first sample
    
    # File I/O
    basis_file_name::String                 # Base filename for basis storage
    state_file_name::String                 # Filename for state storage
    
    function IncrementalSVD(options::SVDOptions, basis_file_name::String, dim::Int)
        # Initialize MPI information
        if MPI.Initialized()
            comm_size = MPI.Comm_size(MPI.COMM_WORLD)
            comm_rank = MPI.Comm_rank(MPI.COMM_WORLD)
        else
            comm_size = 1
            comm_rank = 0
        end
        
        # Gather dimensions from all processes
        proc_dims = Vector{Int}(undef, comm_size)
        if MPI.Initialized()
            MPI.Allgather!([dim], proc_dims, MPI.COMM_WORLD)
        else
            proc_dims[1] = dim
        end
        
        total_dim = sum(proc_dims)
        
        # Create state filename
        state_file_name = ""
        if options.save_state || options.restore_state
            state_file_name = @sprintf("%s.state.%06d.h5", basis_file_name, comm_rank)
        end
        
        svd_obj = new(
            nothing, nothing, nothing, nothing, nothing,
            dim, options.linearity_tol, options.skip_linearly_dependent,
            options.max_basis_dimension, options.save_state, options.update_right_SV,
            options.debug_algorithm,
            comm_size, comm_rank, proc_dims, total_dim,
            0, 1, true,
            basis_file_name, state_file_name
        )
        
        # Restore state if requested
        if options.restore_state
            restore_state!(svd_obj)
        end
        
        return svd_obj
    end
end

"""
    restore_state!(svd_obj::IncrementalSVD)

Restore SVD state from HDF5 file.
"""
function restore_state!(svd_obj::IncrementalSVD)
    if !isfile(svd_obj.state_file_name)
        @warn "State file $(svd_obj.state_file_name) not found, starting fresh"
        return false
    end
    
    try
        h5open(svd_obj.state_file_name, "r") do file
            # Read U matrix
            if haskey(file, "U")
                svd_obj.U = read(file, "U")
                svd_obj.num_samples = size(svd_obj.U, 2)
            end
            
            # Read S vector
            if haskey(file, "S")
                svd_obj.S = read(file, "S")
            end
            
            # Read W matrix if updating right singular vectors
            if svd_obj.update_right_SV && haskey(file, "W")
                svd_obj.W = read(file, "W")
                svd_obj.num_rows_of_W = size(svd_obj.W, 1)
            end
        end
        
        svd_obj.is_first_sample = false
        return true
    catch e
        @warn "Failed to restore state: $e"
        return false
    end
end

"""
    save_state!(svd_obj::IncrementalSVD)

Save SVD state to HDF5 file.
"""
function save_state!(svd_obj::IncrementalSVD)
    if !svd_obj.save_state || svd_obj.is_first_sample
        return
    end
    
    try
        h5open(svd_obj.state_file_name, "w") do file
            if svd_obj.U !== nothing
                write(file, "U", svd_obj.U)
            end
            
            if svd_obj.S !== nothing
                write(file, "S", svd_obj.S)
            end
            
            if svd_obj.update_right_SV && svd_obj.W !== nothing
                write(file, "W", svd_obj.W)
            end
        end
    catch e
        @warn "Failed to save state: $e"
    end
end

"""
    get_spatial_basis(svd_obj::AbstractIncrementalSVD)

Returns the spatial basis vectors (left singular vectors).
"""
function get_spatial_basis(svd_obj::IncrementalSVD)
    @assert svd_obj.basis !== nothing "Basis not computed"
    return svd_obj.basis
end

"""
    get_temporal_basis(svd_obj::AbstractIncrementalSVD)

Returns the temporal basis vectors (right singular vectors).
"""
function get_temporal_basis(svd_obj::IncrementalSVD)
    @assert svd_obj.basis_right !== nothing "Temporal basis not computed"
    return svd_obj.basis_right
end

"""
    get_singular_values(svd_obj::AbstractIncrementalSVD)

Returns the singular values.
"""
function get_singular_values(svd_obj::IncrementalSVD)
    @assert svd_obj.S !== nothing "Singular values not computed"
    return svd_obj.S
end

"""
    construct_Q(svd_obj::IncrementalSVD, l::Vector{Float64}, k::Float64)

Construct the Q matrix [diag(S), l; 0', k] for SVD computation.
"""
function construct_Q(svd_obj::IncrementalSVD, l::Vector{Float64}, k::Float64)
    @assert length(l) == svd_obj.num_samples "Vector l dimension mismatch"
    
    n = svd_obj.num_samples + 1
    Q = zeros(n, n)
    
    # Fill diagonal with singular values
    for i in 1:svd_obj.num_samples
        Q[i, i] = svd_obj.S[i]
    end
    
    # Fill last column with l
    for i in 1:svd_obj.num_samples
        Q[i, n] = l[i]
    end
    
    # Set bottom-right element
    Q[n, n] = k
    
    return Q
end

# Dispatch functions for different algorithm types
function build_initial_svd!(svd::IncrementalSVD, u::Vector{Float64})
    error("build_initial_svd! not implemented for base IncrementalSVD type. Use a concrete implementation like IncrementalSVDBrand, IncrementalSVDFastUpdate, or IncrementalSVDStandard.")
end

function compute_basis!(svd::IncrementalSVD)
    error("compute_basis! not implemented for base IncrementalSVD type. Use a concrete implementation like IncrementalSVDBrand, IncrementalSVDFastUpdate, or IncrementalSVDStandard.")
end

function add_linearly_dependent_sample!(svd::IncrementalSVD, A::Matrix{Float64}, W::Matrix{Float64}, sigma::Matrix{Float64})
    error("add_linearly_dependent_sample! not implemented for base IncrementalSVD type. Use a concrete implementation like IncrementalSVDBrand, IncrementalSVDFastUpdate, or IncrementalSVDStandard.")
end

function add_new_sample!(svd::IncrementalSVD, j::Vector{Float64}, A::Matrix{Float64}, W::Matrix{Float64}, sigma::Matrix{Float64})
    error("add_new_sample! not implemented for base IncrementalSVD type. Use a concrete implementation like IncrementalSVDBrand, IncrementalSVDFastUpdate, or IncrementalSVDStandard.")
end

# Fixed base.jl - The key changes for proper parallelization

"""
    take_sample!(svd_obj::IncrementalSVD, u_local::Vector{Float64}; add_without_increase::Bool=false)

Sample new state vector u_local (local portion on this processor). Returns true if sampling was successful.
Note: u_local should be the LOCAL portion of the distributed vector on this processor.
"""
function take_sample!(svd_obj::IncrementalSVD, u_local::Vector{Float64}; add_without_increase::Bool=false)
    @assert length(u_local) == svd_obj.dim "Input vector dimension mismatch - expected $(svd_obj.dim), got $(length(u_local))"
    
    # Check if input is non-zero (requires global reduction)
    local_norm_sq = dot(u_local, u_local)
    global_norm_sq = local_norm_sq
    if MPI.Initialized() && svd_obj.comm_size > 1
        global_norm_sq = MPI.Allreduce(local_norm_sq, MPI.SUM, MPI.COMM_WORLD)
    end
    
    if sqrt(global_norm_sq) ≈ 0.0
        return false
    end
    
    # Build initial SVD or add incremental sample
    if svd_obj.is_first_sample
        build_initial_svd!(svd_obj, u_local)
        svd_obj.is_first_sample = false
    else
        result = build_incremental_svd!(svd_obj, u_local, add_without_increase)
        if !result
            return false
        end
    end
    
    # Debug output
    if svd_obj.debug_algorithm
        debug_output(svd_obj)
    end
    
    return true
end

"""
    build_incremental_svd!(svd_obj::IncrementalSVD, u_local::Vector{Float64}, add_without_increase::Bool)

Core incremental SVD algorithm. Adds new sample to existing decomposition.
u_local is the LOCAL portion of the distributed vector.
"""
function build_incremental_svd!(svd_obj::IncrementalSVD, u_local::Vector{Float64}, add_without_increase::Bool)
    # Compute projection: l = U' * u (requires global communication)
    l_local = svd_obj.U' * u_local  # Local contribution
    l = copy(l_local)
    if MPI.Initialized() && svd_obj.comm_size > 1
        # Sum contributions from all processors
        MPI.Allreduce!(l, MPI.SUM, MPI.COMM_WORLD)
    end
    
    # Compute basis projection: basis_l = U * l (local computation)
    basis_l_local = svd_obj.U * l
    
    # Compute residual more accurately to avoid catastrophic cancellation
    e_proj_local = u_local - basis_l_local
    
    # Compute global norm of residual (requires global communication)
    k_local_sq = dot(e_proj_local, e_proj_local)
    k_global_sq = k_local_sq
    if MPI.Initialized() && svd_obj.comm_size > 1
        k_global_sq = MPI.Allreduce(k_local_sq, MPI.SUM, MPI.COMM_WORLD)
    end
    
    k = sqrt(k_global_sq)
    
    if k <= 0
        if svd_obj.comm_rank == 0
            println("Linearly dependent sample detected!")
        end
        k = 0.0
    end
    
    # Determine if sample is linearly dependent
    linearly_dependent = false
    
    if k < svd_obj.linearity_tol
        if svd_obj.comm_rank == 0
            println("Linearly dependent sample! k = $k")
            println("linearity_tol = $(svd_obj.linearity_tol)")
        end
        k = 0.0
        linearly_dependent = true
    elseif svd_obj.num_samples >= svd_obj.max_basis_dimension || add_without_increase
        k = 0.0
        linearly_dependent = true
    elseif svd_obj.num_samples >= svd_obj.total_dim
        linearly_dependent = true
    end
    
    # Construct Q matrix for SVD (only on rank 0, or replicated)
    Q = construct_Q(svd_obj, l, k)
    
    # Perform SVD of Q (can be done on all processors since Q is small)
    try
        F = svd(Q)
        U_q, S_q, V_q = F.U, F.S, F.Vt'
        
        # Convert to matrices for consistency
        A = Matrix(U_q)
        sigma = diagm(S_q)
        W = Matrix(V_q)
        
        # Add sample based on linear dependence
        if linearly_dependent && !svd_obj.skip_linearly_dependent
            if svd_obj.comm_rank == 0
                println("Adding linearly dependent sample!")
            end
            add_linearly_dependent_sample!(svd_obj, A, W, sigma)
        elseif !linearly_dependent
            # Compute normalized residual direction (local portion)
            j_local = e_proj_local / k
            add_new_sample!(svd_obj, j_local, A, W, sigma)
        end
        
        # Compute basis vectors
        compute_basis!(svd_obj)
        
        return true
    catch e
        @warn "SVD computation failed: $e"
        return false
    end
end

# Updated debug output for parallel case
"""
    debug_output(svd_obj::IncrementalSVD)

Print debug information about current SVD state (handles distributed data).
"""
function debug_output(svd_obj::IncrementalSVD)
    if svd_obj.comm_rank == 0 && svd_obj.basis !== nothing
        println("Singular values:")
        for i in 1:svd_obj.num_samples
            @printf("%.16e\n", svd_obj.S[i])
        end
        println()
        
        println("Spatial basis (rank 0 portion):")
        for i in 1:min(size(svd_obj.basis, 1), 10)  # Show only first 10 rows
            for j in 1:svd_obj.num_samples
                @printf("%.6e ", svd_obj.basis[i, j])
            end
            println()
        end
        
        # Gather and print other processors' portions
        if svd_obj.comm_size > 1
            for proc in 1:(svd_obj.comm_size-1)
                println("Spatial basis (rank $proc portion):")
                # Note: In a real implementation, you'd use MPI.Recv here
                # For now, we'll skip this to avoid complexity
                println("  [Data from rank $proc - dimensions: $(svd_obj.proc_dims[proc+1])]")
            end
        end
        
        println("=" ^ 60)
    end
end