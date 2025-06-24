"""
    create_incremental_svd_brand(; kwargs...)

Create IncrementalSVDBrand with default options.
"""
function create_incremental_svd_brand(;
    dim::Int,
    basis_file_name::String = "basis",
    linearity_tol::Float64 = 1e-12,
    singular_value_tol::Float64 = 0.0,
    skip_linearly_dependent::Bool = false,
    max_basis_dimension::Int = typemax(Int),
    save_state::Bool = false,
    restore_state::Bool = false,
    update_right_SV::Bool = false,
    debug_algorithm::Bool = false
)
    options = SVDOptions(
        linearity_tol = linearity_tol,
        singular_value_tol = singular_value_tol,
        skip_linearly_dependent = skip_linearly_dependent,
        max_basis_dimension = max_basis_dimension,
        save_state = save_state,
        restore_state = restore_state,
        update_right_SV = update_right_SV,
        debug_algorithm = debug_algorithm
    )
    
    return IncrementalSVDBrand(options, basis_file_name, dim)
end

"""
    create_incremental_svd_fast_update(; kwargs...)

Create IncrementalSVDFastUpdate with default options.
"""
function create_incremental_svd_fast_update(;
    dim::Int,
    basis_file_name::String = "basis",
    linearity_tol::Float64 = 1e-12,
    singular_value_tol::Float64 = 0.0,
    skip_linearly_dependent::Bool = false,
    max_basis_dimension::Int = typemax(Int),
    save_state::Bool = false,
    restore_state::Bool = false,
    update_right_SV::Bool = false,
    debug_algorithm::Bool = false
)
    options = SVDOptions(
        linearity_tol = linearity_tol,
        singular_value_tol = singular_value_tol,
        skip_linearly_dependent = skip_linearly_dependent,
        max_basis_dimension = max_basis_dimension,
        save_state = save_state,
        restore_state = restore_state,
        update_right_SV = update_right_SV,
        debug_algorithm = debug_algorithm
    )
    
    return IncrementalSVDFastUpdate(options, basis_file_name, dim)
end

"""
    create_incremental_svd_standard(; kwargs...)

Create IncrementalSVDStandard with default options.
"""
function create_incremental_svd_standard(;
    dim::Int,
    basis_file_name::String = "basis",
    linearity_tol::Float64 = 1e-12,
    skip_linearly_dependent::Bool = false,
    max_basis_dimension::Int = typemax(Int),
    save_state::Bool = false,
    restore_state::Bool = false,
    update_right_SV::Bool = false,
    debug_algorithm::Bool = false
)
    options = SVDOptions(
        linearity_tol = linearity_tol,
        singular_value_tol = 0.0,  # Not used in standard algorithm
        skip_linearly_dependent = skip_linearly_dependent,
        max_basis_dimension = max_basis_dimension,
        save_state = save_state,
        restore_state = restore_state,
        update_right_SV = update_right_SV,
        debug_algorithm = debug_algorithm
    )
    
    return IncrementalSVDStandard(options, basis_file_name, dim)
end
