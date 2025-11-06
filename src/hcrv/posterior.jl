# export functions
export posterior_gamma_mcmc, posterior_gamma_exact

"""
    posterior_gamma_mcmc(X, alpha0, rate0, b, num_samples; kwargs...)

Posterior sampling from the gamma-gamma hierarchical CRV, using MCMC algorithms

# Arguments:
- `X`: vectors of observations for each population
- `alpha0`: model parameter
- `rate0`: model parameter (b0 / alpha)
- `b`: model parameter (relevant only if norm = false)
- `num_samples`: number of output posterior samples

# Keyword Arguments:
- `burnin = 0`: burnin steps
- `thin = 1`: thinning lag
- `L = 0`: number of jumps from the continuous component (if L = 0, total mass)
- `normalize = false`: returns normalized probability weights (true) or unnormalized jumps (false)
- `logscale = true`: proposes MH steps using Gamma proposals (false) or logNormal proposals (true)

# Returns:
- `jumps`: posterior samples from jumps at fixed locations
- `jumpsc`: posterior samples from jumps at random locations
- `counts`: number of observations for each population and distinct value
- `Xstar`: distinct values
- `dgn`: DiagnosticsMCMC object
"""
function posterior_gamma_mcmc(X::Vector{Vector{T}}, alpha0::Float64, rate0::Float64, b::Float64, num_samples::Int64;
        burnin::Int64 = 0, thin::Int64 = 1, L::Int64 = 0, normalize::Bool = false, logscale::Bool = true) where T
    
    # extract info from data
    counts, Xstar = setup_hcrv(X)
    
    # sample jumps
    jumps, jumpsc, dgn = sample_posterior_mcmc(counts, alpha0, rate0, b, num_samples, burnin, thin, L, normalize, logscale)
    
    return jumps, jumpsc, counts, Xstar, dgn

end # posterior_gamma_mcmc

"""
    posterior_gamma_exact(X, alpha0, rate0, b, num_samples; kwargs...)

Posterior sampling from the gamma-gamma hierarchical CRV, using exact sampling algorithms

# Arguments:
- `X`: vectors of observations for each population
- `alpha0`: model parameter
- `rate0`: model parameter (b0 / alpha)
- `b`: model parameter (relevant only if norm = false)
- `num_samples`: number of output posterior samples

# Keyword Arguments:
- `L = 0`: number of jumps from the continuous component (if L = 0, total mass)
- `normalize = false`: returns normalized probability weights (true) or unnormalized jumps (false)

# Returns:
- `jumps`: posterior samples from jumps at fixed locations
- `jumpsc`: posterior samples from jumps at random locations
- `counts`: number of observations for each population and distinct value
- `Xstar`: distinct values
- `dgn`: DiagnosticsExact object
"""
function posterior_gamma_exact(X::Vector{Vector{T}}, alpha0::Float64, rate0::Float64, b::Float64, num_samples::Int64; 
        L::Int = 0, normalize::Bool = false) where T
    
    # extract info from data
    counts, Xstar = setup_hcrv(X)
    
    # sample jumps
    try
        jumps, jumpsc, dgn = sample_posterior_exact(counts, alpha0, rate0, b, num_samples, L, normalize)
        return jumps, jumpsc, counts, Xstar, dgn
    catch error
        @warn string(error)
        return zeros(0, 0, 0), zeros(0, 0, 0), counts, Xstar, nothing
    end

end # posterior_gamma_exact

"""
    setup_hcrv(X)

"""
function setup_hcrv(X::Vector{Vector{T}}) where T

    # number of groups
    d = length(X)
    
    # vector of observations
    allX = vcat(X...)
    
    # find unique values
    Xstar = sort(unique(allX))
    k = length(Xstar)
    
    # count occurrences for each group and value
    counts = zeros(Int64, d, k)
    for (i, X_group) in enumerate(X)
        for (j, value) in enumerate(Xstar)
            counts[i,j] = sum(X_group .== value)
        end
    end
    
    return counts, Xstar

end # setup_hcrv

"""
    sample_posterior_mcmc(counts, alpha0, rate0, b, num_samples, burnin, thin, L, normalize, logscale)

"""
function sample_posterior_mcmc(counts::Matrix{Int64}, alpha0::Float64, rate0::Float64, b::Float64, num_samples::Int64, burnin::Int64, thin::Int64, L::Int64, normalize::Bool, logscale::Bool)

    # retrieve dimensions
    d, k = size(counts)
    
    # compute counts for groups
    counts_group = vec(sum(counts, dims = 2))
    
    # initialize diagnostics
    dgn = DiagnosticsMCMC(k)
    
    # initialize state
    baselatents, basejumps = ones(k+1), ones(k)
    
    # initialize variance parameters
    varpar_latent, varpars_jumps = 1.0, ones(k)
    
    # burnin
    for s in range(1, burnin)
        
        # sample baselatent
        accept = sample_baselatents_mcmc(baselatents, counts, rate0, alpha0, logscale, varpar_latent)
        varpar_latent *= acceptance_tuning(accept, s)
        
        # sample latent variables
        latents = sample_latents(counts_group, sum(baselatents))
        
        # precompute rate
        rate = rate0 + sum(log.(1.0 .+ latents))
        
        # sample basejumps
        for j in range(1,k)
            basejumps[j], accept = sample_basejump_mcmc(basejumps[j], counts[:,j], rate, logscale, varpars_jumps[j])
            varpars_jumps[j] *= acceptance_tuning(accept, s)
        end

    end
    
    # compute burnin time
    dgn.time_burnin = mytime() - dgn.etime

    # initialize output
    jumps = Array{Float64,3}(undef, k, d, num_samples)
    jumpsc = Array{Float64,3}(undef, L + (L == 0 ? 1 : 0), d, num_samples)
    
    # initialize auxiliary variables
    accept_latent, accept_jumps = 0.0, zeros(k)
    latents, rate = zeros(k+1), 0.0
    
    # posterior sampling
    for s in range(1, num_samples)

        # thinning
        for _ in 1:thin
            
            # sample baselatent
            accept_latent = sample_baselatents_mcmc(baselatents, counts, rate0, alpha0, logscale, varpar_latent)
            
            # sample latent variables
            latents = sample_latents(counts_group, sum(baselatents))
            
            # precompute rate
            rate = rate0 + sum(log.(1.0 .+ latents))
            
            # sample basejumps
            for j in range(1,k)
                basejumps[j], accept_jumps[j] = sample_basejump_mcmc(basejumps[j], counts[:,j], rate, logscale, varpars_jumps[j])
            end

        end
        
        # update acceptance rates
        dgn.accept_latent += accept_latent
        dgn.accept_jumps += accept_jumps
        
        # sample from basecrm
        basejumpsc = gamma_process(alpha0, rate, L)
        
        if !normalize
            
            # sample jumps
            jumps[:,:,s] = sample_jumps(counts, basejumps, latents, b)
            
            # sample jumps from crms
            jumpsc[:,:,s] = sample_jumps_crm(basejumpsc, latents, b)

        else  # normalized jumps

            jumps[:,:,s], jumpsc[:,:,s] = sample_normalized_jumps(counts, basejumps, basejumpsc)

        end

    end
    
    # rescale acceptance rates
    dgn.accept_latent /= num_samples
    dgn.accept_jumps ./= num_samples
    
    # compute execution time
    dgn.etime = mytime() - dgn.etime
    
    return jumps, jumpsc, dgn

end # sample_posterior_mcmc

"""
    sample_posterior_exact(counts, alpha0, rate0, b, num_samples, L, normalize)

"""
function sample_posterior_exact(counts::Matrix{Int64}, alpha0::Float64, rate0::Float64, b::Float64, num_samples::Int64, L::Int64, normalize::Bool)

    # retrieve dimensions
    d, k = size(counts)
    
    # compute counts for groups
    counts_group = vec(sum(counts, dims = 2))
    
    # Initialize diagnostics
    dgn = DiagnosticsExact()
    
    # compute coefficients
    weights_jumps = compute_weights_jumps(counts)
    weights = compute_weights_latent(weights_jumps)

    # compute initialization time
    dgn.time_init = mytime() - dgn.etime
    
    # sample baselatent
    # if method == 0
        baselatent_samples, dgn.accept_latent, _, _ = sample_baselatent_rejection(weights, counts_group, rate0, alpha0, num_samples)
    # elseif method == 1
        # baselatent_samples, dgn.accept_latent, _, _ = sample_baselatent_ars(weights, counts_group, rate0, alpha0, num_samples)
    # end
    
    # compute baselatent sampling time
    dgn.time_latent = mytime() - dgn.etime

    # initialize output
    jumps = Array{Float64,3}(undef, k, d, num_samples)
    jumpsc = Array{Float64,3}(undef, L + (L == 0 ? 1 : 0), d, num_samples)

    # initialize basejumps
    basejumps = ones(k)
    
    for (s, baselatent) in enumerate(baselatent_samples)

        # sample latent variables
        latents = sample_latents(counts_group, baselatent)
        
        # precompute rate
        rate = rate0 + sum(log.(1.0 .+ latents))
        
        # sample basejumps
        for (j, weightsj) in enumerate(weights_jumps)
            basejumps[j] = sample_basejump_exact(weightsj, rate)
        end
        
        # sample from basecrm
        basejumpsc = gamma_process(alpha0, rate, L)
        
        if !normalize
            
            # sample jumps
            jumps[:,:,s] = sample_jumps(counts, basejumps, latents, b)
            
            # sample jumps from crms
            jumpsc[:,:,s] = sample_jumps_crm(basejumpsc, latents, b)

        else  # normalized jumps

            jumps[:,:,s], jumpsc[:,:,s] = sample_normalized_jumps(counts, basejumps, basejumpsc)

        end

    end
    
    # compute execution time
    dgn.etime = mytime() - dgn.etime
    
    return jumps, jumpsc, dgn

end # sample_posterior_exact
