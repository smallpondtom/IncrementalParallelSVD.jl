# Cleanup function for proper state saving
"""
    finalize!(svd::AbstractIncrementalSVD)

Finalize SVD computation and save state if requested.
"""
function finalize!(svd::IncrementalSVDBrand)
    save_state!(svd.base)
    save_brand_state!(svd)
end

function finalize!(svd::IncrementalSVDFastUpdate)
    save_state!(svd.base)
    # FastUpdate uses same state format as Brand
    if svd.base.save_state && !svd.base.is_first_sample
        try
            h5open(svd.base.state_file_name, "r+") do file
                if svd.Up !== nothing
                    write(file, "Up", svd.Up)
                end
            end
        catch e
            @warn "Failed to save FastUpdate state: $e"
        end
    end
end

function finalize!(svd::IncrementalSVDStandard)
    save_state!(svd.base)
end
