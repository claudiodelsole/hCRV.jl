# imports
import Random: seed!
import Distributions: Poisson, Categorical, Beta, Dirichlet

"""
    model_poisson(counts_group, means; seed = 0)

Generate independent observations from Poisson distributions.

# Arguments:
- `counts_group`: number of observations for each population
- `means`: Poisson parameters for each population
- `seed`: seed for reproducibility (if seed = 0, no seed is set)

# Returns:
- `X`: observations for each population
"""
function model_poisson(counts_group::Vector{Int64}, means::Vector{Float64}; seed::Int64 = 0)::Vector{Vector{Int64}}

    # check input
    @assert length(counts_group) == length(means)
    
    # set seed
    if seed > 0 seed!(seed) end
    
    # initialize output
    X = Vector{Vector{Int64}}(undef, 0)
    
    # sample observations
    for (mu, numobs) in zip(means, counts_group)
        X_group = rand(Poisson(mu), numobs)
        push!(X, X_group)
    end
    
    return X

end # model_poisson

"""
    model_hdp(counts_group, alpha0, alpha; hyperprior = false, L = 100, seed = 0)

Generate observations from hierarchical Dirichlet process.

# Arguments:
- `counts_group`: number of observations for each population
- `alpha0`: concentration parameter for root measure
- `alpha`: concentration parameter

# Keyword Arguments:
- `hyperprior = false`: gamma hyperprior on concentration parameter (to match gamma-gamma hierarchical CRV model)
- `L = 100`: number of jumps from the root measure
- `seed = 0`: seed for reproducibility (if seed = 0, no seed is set)

# Returns:
- `X`: observations for each population
"""
function model_hdp(counts_group::Vector{Int64}, alpha::Float64, alpha0::Float64; 
        hyperprior::Bool = false, L::Int64 = 100, seed::Int64 = 0)::Vector{Vector{Int64}}
    
    # set seed
    if seed > 0 seed!(seed) end
    
    # sample concentration parameter
    if hyperprior alpha = rand(Gamma(alpha0)) / alpha end
    
    # sample baseprobs
    baseprobs = stick_breaking(alpha0, L)
    
    # initialize output
    X = Vector{Vector{Int64}}(undef, 0)
    
    # loop on groups
    for numobs in counts_group
        
        # sample from Dirichlet distribution
        idxs = baseprobs .> 0
        probs = rand(Dirichlet(alpha * baseprobs[idxs]))
        
        # sample observations
        X_group = rand(Categorical(probs), numobs)
        push!(X, X_group)

    end
    
    return X

end # model_hdp

"""
    stick_breaking(alpha, L)

"""
function stick_breaking(alpha::Float64, L::Int64)::Vector{Float64}
    
    # total mass
    if L == 0 return [1.0] end
    
    # initialize
    probs = Vector{Float64}(undef, L)
    ratio, pratios = 0.0, 1.0
    
    # stick-breaking ratios
    for h in eachindex(probs)
        ratio = rand(Beta(1.0, alpha))
        probs[h] = ratio * pratios
        pratios *= (1.0 - ratio)
    end
    
    return probs

end # stick_breaking
