"""
   compute_stirling_coeffs(counts)

"""
function compute_stirling_coeffs(counts::Matrix{Int64})::Tuple{Vector{Vector{Float64}}, Vector{Int64}}
    
    # initialize coefficients
    stirling_coeffs = Vector{Vector{Float64}}(undef, 0)
    mincounts = Vector{Int64}(undef, 0)
    
    # loop on clusters
    for countsj in eachcol(counts)
        
        # compute coefficients (starting from index 1)
        coeffsj = generalized_stirling_numbers(Vector(countsj))
        
        # normalize coefficients
        normalize_coeffs!(coeffsj, use_min = false)
        
        # rescale coefficients
        mincountsj, cumprod = findfirst(x -> x != 0.0, coeffsj), 1.0
        for h in range(mincountsj, length(coeffsj))
            coeffsj[h] *= cumprod
            cumprod *= h
        end
        
        # # normalize coefficients
        normalize_coeffs!(coeffsj)
        
        # append coefficients
        push!(stirling_coeffs, coeffsj)
        push!(mincounts, mincountsj)

    end
    
    return stirling_coeffs, mincounts

end # compute_stirling_coeffs

"""
    normalize_coeffs!(coeffs; use_min = true)

"""
function normalize_coeffs!(coeffs::Vector{Float64}; use_min::Bool = true)
    
    # check coefficients
    if isinf(sum(coeffs))
        error("Cannot compute coefficients")
    end
    
    # normalizing constant
    positive_coeffs = coeffs[coeffs .> 0.0]
    normconst = use_min ? minimum(positive_coeffs) : maximum(coeffs)
    
    # normalize coefficients
    mincountsj = findfirst(x -> x != 0.0, coeffs)
    coeffs[mincountsj:end] ./= normconst

end # normalize_coeffs!
