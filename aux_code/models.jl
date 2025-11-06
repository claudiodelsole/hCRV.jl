# imports
import Random: seed!
import Distributions: Normal, Poisson, Categorical, Beta, Dirichlet

"""
    model_poisson(counts_group, means; seed = 0)

"""
function model_poisson(counts_group::Vector{Int64}, means::Vector{Float64}; seed::Int64 = 0)
    
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
    model_hdp(counts_group, alpha, alpha0; hyperprior = false, L = 100, seed = 0)

"""
function model_hdp(counts_group::Vector{Int64}, alpha::Float64, alpha0::Float64; 
        hyperprior::Bool = false, L::Int64 = 100, seed::Int64 = 0)
    
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
    model_mixture(counts_group, centers, stdevs, probs; seed = 0)

"""
function model_mixture(counts_group::Vector{Int64}, centers::Vector{Float64}, stdevs::Vector{Float64}, probs::Matrix{Float64}; seed::Int64 = 0)

    # set seed
    if seed > 0 seed!(seed) end
    
    # initialize output
    X = Vector{Vector{Float64}}(undef, 0)
    
    # sample observations
    for (i, numobs) in enumerate(counts_group)
        clusters = rand(Categorical(probs[i,:]), numobs)
        X_group = rand(Normal(centers[clusters], stdevs[clusters]))
        push!(X, X_group)
    end
    
    return X

end # model_mixture

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
