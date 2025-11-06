"""
   compute_weights_jumps(counts)

"""
function compute_weights_jumps(counts::Matrix{Int64})::Vector{Vector{Float64}}
    
    # initialize weights
    weights_jumps = Vector{Vector{Float64}}(undef, 0)
    
    # loop on clusters
    for countsj in eachcol(counts)
        
        # compute weights (starting from index 1)
        weightsj = generalized_stirling_numbers(Vector(countsj))
        
        # normalize weights
        normalize_weights!(weightsj, use_min = false)
        
        # rescale weights
        mincountsj, cumprod = findfirst(x -> x != 0.0, weightsj), 1.0
        for h in range(mincountsj, length(weightsj))
            weightsj[h] *= cumprod
            cumprod *= h
        end
        
        # normalize weights
        normalize_weights!(weightsj)
        
        # append weights
        push!(weights_jumps, weightsj)

    end
    
    return weights_jumps

end # compute_weights_jumps

"""
    compute_weights_latent(weights_jumps)

"""
function compute_weights_latent(weights_jumps::Vector{Vector{Float64}})::Vector{Float64}
    
    # initialize output
    weights = [1.0]
    
    # compute weights
    for weightsj in weights_jumps
        
        # convolution
        weights = conv(weights, weightsj)
        
        # normalize weights
        normalize_weights!(weights)

    end
    
    # concatenate to start from index 1
    weights = vcat(zeros(length(weights_jumps)-1), weights)
    
    return weights

end # compute_weights_latent

"""
    normalize_weights!(weights; use_min = true)

"""
function normalize_weights!(weights::Vector{Float64}; use_min::Bool = true)
    
    # check weights
    if isinf(sum(weights))
        error("Cannot compute weights!")
    end
    
    # normalizing constant
    positive_weights = weights[weights .> 0.0]
    normconst = use_min ? minimum(positive_weights) : maximum(weights)
    
    # normalize weights
    mincountsj = findfirst(x -> x != 0.0, weights)
    weights[mincountsj:end] ./= normconst

end # normalize_weights!
