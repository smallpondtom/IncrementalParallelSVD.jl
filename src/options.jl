# Options structure for SVD configuration
struct SVDOptions
    linearity_tol::Float64
    singular_value_tol::Float64
    skip_linearly_dependent::Bool
    max_basis_dimension::Int
    save_state::Bool
    restore_state::Bool
    update_right_SV::Bool
    debug_algorithm::Bool
    
    function SVDOptions(;
        linearity_tol::Float64 = 1e-12,
        singular_value_tol::Float64 = 0.0,
        skip_linearly_dependent::Bool = false,
        max_basis_dimension::Int = typemax(Int),
        save_state::Bool = false,
        restore_state::Bool = false,
        update_right_SV::Bool = false,
        debug_algorithm::Bool = false
    )
        @assert linearity_tol > 0.0 "linearity_tol must be positive"
        @assert singular_value_tol >= 0.0 "singular_value_tol must be non-negative"
        @assert max_basis_dimension > 0 "max_basis_dimension must be positive"
        
        new(linearity_tol, singular_value_tol, skip_linearly_dependent,
            max_basis_dimension, save_state, restore_state, update_right_SV,
            debug_algorithm)
    end
end
