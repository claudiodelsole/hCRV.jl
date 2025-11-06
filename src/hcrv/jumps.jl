"""
    sample_basejump_mcmc(jump, countsj, rate, logscale, varpar)

"""
function sample_basejump_mcmc(jump::Float64, countsj::Vector{Int64}, rate::Float64, logscale::Bool, varpar::Float64)::Tuple{Float64, Float64}
    
    # compute delta
    delta = 1.0 / varpar
    
    if maximum(countsj) == 1    # special case
        return rand(Gamma(sum(countsj))) / rate, NaN
    end
    
    # propose perturbation
    eps = logscale ? exp(sqrt(varpar) * randn()) : rand(Gamma(delta)) / delta
    
    # compute acceptance probability
    logaccept = loglikelihood_basejump(jump * eps, countsj, rate)
    logaccept -= loglikelihood_basejump(jump, countsj, rate)
    if !logscale logaccept += delta * (eps - 1.0 / eps) - 2.0 * delta * log(eps) end
    
    # accept / reject
    if log(rand()) < logaccept
        jump *= eps
    end
    
    return jump, min(exp(logaccept), 1.0)

end # sample_basejump_mcmc

"""
    loglikelihood_basejump(t, countsj, rate)

"""
function loglikelihood_basejump(t::Float64, countsj::Vector{Int64}, rate::Float64)::Float64
    
    # compute loglikelihood
    logpdf = - rate * t
    
    # rising factorials
    for count in countsj
        logpdf += rising_factorial(t, count)
    end
    
    return logpdf

end # loglikelihood_basejump

"""
    sample_basejump_exact(weightsj, rate)

"""
function sample_basejump_exact(weightsj::Vector{Float64}, rate::Float64)::Float64
    
    # initialize probability weights
    masses = copy(weightsj)
    
    # compute probability weights
    mincountsj, prate = findfirst(x -> x != 0, weightsj), 1.0
    for h in range(mincountsj, length(weightsj))
        masses[h] /= prate
        prate *= rate
    end
    
    # sample shape parameter
    shape = sample_categorical(masses[mincountsj:end]) + mincountsj - 1
    
    # sample from gamma distribution
    return rand(Gamma(shape)) / rate

end # sample_basejump_exact

"""
    logdensity_basejump(t, countsj, rate)

"""
function logdensity_basejump(t::Float64, countsj::Vector{Int64}, rate::Float64)::Float64
    
    # compute logdensity
    logpdf = - rate * t - log(t)
    
    # rising factorials
    for count in countsj
        logpdf += rising_factorial(t, count)
    end
    
    return logpdf

end # logdensity_basejump

"""
    sample_jumps(counts, basejumps, latents, b)

"""
function sample_jumps(counts::Matrix{Int64}, basejumps::Vector{Float64}, latents::Vector{Float64}, b::Float64)::Matrix{Float64}

    # initialize output
    jumps = zeros(length(basejumps), length(latents))
    
    # sample from gamma distribution
    for (j, jump) in enumerate(basejumps)
        for (i, latent) in enumerate(latents)
            jumps[j,i] = rand(Gamma(counts[i,j] + jump)) / (b + b * latent)
        end
    end
    
    return jumps

end # sample_jumps

"""
    sample_normalized_jumps(counts, basejumps, basejumpsc)

"""
function sample_normalized_jumps(counts::Matrix{Int64}, basejumps::Vector{Float64}, basejumpsc::Vector{Float64})::Tuple{Matrix{Float64}, Matrix{Float64}}

    # retrieve dimensions
    d, k = size(counts)
    
    # initialize output
    probs, probsc = zeros(length(basejumps), d), zeros(length(basejumpsc), d)
    
    # find indexes for positive jumps
    idxs = basejumpsc .> 0.0
    
    # sample from Dirichlet distribution
    for i in range(1, d)
        pdir = rand(Dirichlet(vcat(counts[i,:] + basejumps, basejumpsc[idxs])))
        probs[:,i], probsc[idxs,i] = pdir[begin:k], pdir[(k+1):end]
    end
    
    return probs, probsc

end # sample_normalized_jumps
