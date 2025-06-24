"""
Random data example.
"""

using MPI
using IncrementalParallelSVD
using LinearAlgebra

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
comm_size = MPI.Comm_size(comm)  # Renamed from 'size' to avoid conflict with size() function

# Set up distributed problem
total_dim = 10000  # Total global dimension
local_dim = div(total_dim, comm_size)  # Local dimension per processor

# Handle remainder for uneven division
if rank < total_dim % comm_size
    local_dim += 1
end

# Create an incremental SVD instance with LOCAL dimension
isvd = create_incremental_svd_brand(
    dim = local_dim,  # This is the LOCAL dimension
    linearity_tol = 1e-12,
    max_basis_dimension = 50,
    update_right_SV = true,
    save_state = false,  # Disable for this example
    debug_algorithm = false
)

println("Rank $rank: local_dim = $local_dim, total_dim = $(isvd.base.total_dim)")

# Generate and distribute samples
n_samples = 20

# Only rank 0 generates the full data and distributes it
if rank == 0
    println("Generating $n_samples samples of dimension $total_dim")
    
    # Generate full global data matrix
    X_global = randn(total_dim, n_samples)
    
    # Process samples one by one
    for i in 1:n_samples
        x_global = X_global[:, i]
        
        # Calculate start and end indices for rank 0's data
        local start_idx = 1
        for r in 0:(rank-1)
            start_idx += div(total_dim, comm_size) + (r < total_dim % comm_size ? 1 : 0)
        end
        local end_idx = start_idx + local_dim - 1
        
        x_local = x_global[start_idx:end_idx]
        
        # Send to other processors
        for dest in 1:(comm_size-1)
            local dest_start = 1
            for r in 0:(dest-1)
                dest_start += div(total_dim, comm_size) + (r < total_dim % comm_size ? 1 : 0)
            end
            dest_local_dim = div(total_dim, comm_size) + (dest < total_dim % comm_size ? 1 : 0)
            local dest_end = dest_start + dest_local_dim - 1
            
            # Send the data array directly
            dest_data = x_global[dest_start:dest_end]
            MPI.Send(dest_data, dest, i, comm)
        end
        
        # Process sample on rank 0
        success = take_sample!(isvd, x_local)
        if !success
            println("Rank $rank: Failed to add sample $i")
        end
    end
    
    # Compare with batch SVD
    U_batch, S_batch, V_batch = svd(X_global)
    
    # Get results from incremental SVD
    spatial_basis_rank0 = get_spatial_basis(isvd)
    temporal_basis = get_temporal_basis(isvd)
    singular_values = get_singular_values(isvd)
    
    println("Rank $rank: Local spatial basis size: $(size(spatial_basis_rank0))")
    
    # Reconstruct full spatial basis by gathering from all processors
    full_spatial_basis = zeros(total_dim, length(singular_values))
    
    # Copy rank 0's portion
    local start_idx = 1
    local end_idx = local_dim
    full_spatial_basis[start_idx:end_idx, :] = spatial_basis_rank0
    
    println("Rank $rank: Placed rank 0 data at indices $start_idx:$end_idx")
    
    # Receive from other processors
    for src in 1:(comm_size-1)
        local src_start = 1
        for r in 0:(src-1)
            src_start += div(total_dim, comm_size) + (r < total_dim % comm_size ? 1 : 0)
        end
        src_local_dim = div(total_dim, comm_size) + (src < total_dim % comm_size ? 1 : 0)
        local src_end = src_start + src_local_dim - 1
        
        println("Rank $rank: Expecting data from rank $src at indices $src_start:$src_end, size ($src_local_dim, $(length(singular_values)))")
        
        # Receive the data array directly, allocating the right size
        src_basis_data = zeros(src_local_dim, length(singular_values))
        MPI.Recv!(src_basis_data, src, 999, comm)
        full_spatial_basis[src_start:src_end, :] = src_basis_data
        
        println("Rank $rank: Received and placed data from rank $src")
    end
    
    # Compute errors
    U_batch_truncated = U_batch[:, 1:length(singular_values)]
    S_batch_truncated = S_batch[1:length(singular_values)]
    
    # Handle sign ambiguity in SVD
    for i in axes(full_spatial_basis, 2)
        if dot(full_spatial_basis[:, i], U_batch_truncated[:, i]) < 0
            full_spatial_basis[:, i] *= -1
        end
    end
    
    sv_error = norm(singular_values - S_batch_truncated)
    basis_error = norm(full_spatial_basis * full_spatial_basis' - U_batch_truncated * U_batch_truncated') / sqrt(2)
    
    println("Rank $rank: Computed $(length(singular_values)) singular values")
    println("Rank $rank: Spatial basis size: $(size(full_spatial_basis))")
    println("Rank $rank: Error in singular values: $sv_error")
    println("Rank $rank: Error in spatial basis: $basis_error")
    
else
    # Non-root processors receive data and process
    for i in 1:n_samples
        # Allocate buffer for receiving data
        x_local = zeros(local_dim)
        MPI.Recv!(x_local, 0, i, comm)
        
        success = take_sample!(isvd, x_local)
        if !success
            println("Rank $rank: Failed to add sample $i")
        end
    end
    
    # Send spatial basis back to rank 0 for error checking
    spatial_basis = get_spatial_basis(isvd)
    
    println("Rank $rank: Computed $(length(get_singular_values(isvd))) singular values")
    println("Rank $rank: Local spatial basis size: $(size(spatial_basis))")
    println("Rank $rank: Sending spatial basis to rank 0")
    
    MPI.Send(spatial_basis, 0, 999, comm)
    
    println("Rank $rank: Successfully sent spatial basis")
end

# Print MPI info for verification
println("Rank $rank: comm_size = $(isvd.base.comm_size)")
println("Rank $rank: proc_dims = $(isvd.base.proc_dims)")

# Clean up
finalize!(isvd)
MPI.Finalize()