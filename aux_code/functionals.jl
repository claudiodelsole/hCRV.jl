# imports
import LinearAlgebra: dot

"""
    random_means(jumps, jumpsc, Xstar, base_samples)

Computation of random means from posterior samples.

# Arguments:
- `jumps`: posterior samples from (normalized) jumps at fixed locations
- `jumpsc`: posterior samples from (normalized) jumps at random locations, total masses if L = 0
- `Xstar`: distinct values
- `base_samples`: samples from the base measure

# Returns:
- `rmeans`: posterior samples from random means
"""
function random_means(jumps::Array{Float64,3}, jumpsc::Array{Float64,3}, Xstar::Vector{Float64}, base_samples::Array{Float64,3})::Matrix{Float64}
    
    # retrieve dimensions
    _, d, num_samples = size(jumps)
    
    # initialize output
    rmeans = Matrix{Float64}(undef, d, num_samples)
    
    # compute means
    for s in axes(rmeans, 2)
        for i in axes(rmeans, 1)
            rmeans[i,s] = dot(Xstar, jumps[:,i,s]) + dot(base_samples[:,i,s], jumpsc[:,i,s])
        end
    end
    
    return rmeans

end # random_means

"""
    random_probsnew(probs, probsc, counts)

Computation of probabilities of unseen values from posterior samples.

# Arguments:
- `probs`: posterior samples from normalized jumps at fixed locations
- `probsc`: posterior samples from normalized jumps at random locations, total masses if L = 0
- `counts`: number of observations for each population and distinct value

# Returns:
- `probsnew`: posterior samples from probabilities of unseen values
"""
function random_probsnew(probs::Array{Float64,3}, probsc::Array{Float64,3}, counts::Matrix{Int})::Matrix{Float64}
    
    # retrieve dimensions
    num_samples, d, _ = size(probs)
    
    # Initialize output
    probsnew = Matrix{Float64}(undef, d, num_samples)
    
    # Compute probabilities
    for s in axes(rmeans, 2)
        for i in axes(rmeans, 1)
            probsnew[i,s] = dot(counts[i,:] .== 0, probs[:,i,s]) + sum(probsc[:,i,s])
        end
    end
    
    return probsnew

end # random_probsnew
