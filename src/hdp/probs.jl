"""
    sample_baseprobs(tables, alpha0)

"""
sample_baseprobs(tables::Vector{Int64}, alpha0::Float64) = rand(Dirichlet(vcat(tables, alpha0)))

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

"""
    sample_probs(counts, baseprobs, baseprobs_dp, alpha)

"""
function sample_probs(counts::Matrix{Int64}, baseprobs::Vector{Float64}, baseprobs_dp::Vector{Float64}, alpha::Float64)

    # retrieve dimensions
    d, k = size(counts)
    
    # initialize output
    probs, probs_hdp = zeros(k, d), zeros(length(baseprobs_dp), d)

    # find indexes for positive jumps
    idxs = (baseprobs_dp .> 0.0)
    
    # sample from dirichlet distribution
    for i in range(1, d)
        pdir = rand(Dirichlet(vcat(counts[i, :] .+ alpha * baseprobs, alpha * baseprobs_dp[idxs])))
        probs[:,i], probs_hdp[idxs,i] = pdir[begin:k], pdir[(k+1):end]
    end

    return probs, probs_hdp

end # sample_probs
