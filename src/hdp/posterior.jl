# export functions
export posterior_hdp

"""
    posterior_hdp(X, alpha0, alpha, num_samples; kwargs...)

Posterior sampling from the hierarchical Dirichlet process, using marginal algorithms (CRF-based or collapsed).

# Arguments:
- `X`: observations for each population
- `alpha0`: concentration parameter for root measure
- `alpha`: concentration parameter
- `num_samples`: number of output posterior samples

# Keyword Arguments:
- `burnin = 0`: burnin steps
- `thin = 1`: thinning lag
- `L = 0`: number of jumps from the continuous component (if L = 0, total mass)
- `prior = false`: impose Gamma prior on the concentration parameter (shape = alpha0, scale = alpha)
- `collapsed = false`: use CRF-base Gibbs sampler (false) or collapsed Gibbs sampler (true)

# Returns:
- `jumps`: posterior samples from jumps at fixed locations
- `jumpsc`: posterior samples from jumps at random locations
- `counts`: number of observations for each population and distinct value
- `Xstar`: distinct values
- `dgn`: [`DiagnosticsHDP`](@ref) object
"""
function posterior_hdp(X::Vector{Vector{T}}, alpha0::Float64, alpha::Float64, num_samples::Int64; 
        burnin::Int64 = 0, thin::Int64 = 1, L::Int64 = 0, prior::Bool = false, collapsed::Bool = false) where T

    if collapsed == false   # marginal restaurant franchise sampler

        # create CRF object
        crf = CRF(X)

        # sample probs
        probs, probsc, dgn = sample_posterior_crf(crf, alpha0, alpha, num_samples, burnin, thin, L, prior, false)

        return probs, probsc, crf.counts, crf.Xstar, dgn

    else    # collapsed Gibbs sampler

        # extract info from data
        counts, Xstar = setup_hcrv(X)

        # sample probs
        try
            probs, probsc, dgn = sample_posterior_collapsed(counts, alpha0, alpha, num_samples, burnin, thin, L, prior, false)
            return probs, probsc, counts, Xstar, dgn
        catch error
            @warn string(error)
            return zeros(0, 0, 0), zeros(0, 0, 0), counts, Xstar, nothing
        end
        
    end

end # posterior_hdp

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
    sample_posterior_crf(crf, alpha0, alpha, num_samples, burnin, thin, L, prior, augmentation)

"""
function sample_posterior_crf(crf::CRF, alpha0::Float64, alpha::Float64, num_samples::Int64, burnin::Int64, thin::Int64, L::Int64, prior::Bool, augmentation::Bool)

    # retrieve dimensions
    d, k = size(crf.counts)

    # initialize diagnostics
    dgn = DiagnosticsHDP()

    # initialize alpha_
    alpha_ = prior ? (alpha * alpha0) : alpha

    # initialize tables
    gibbs_step_initialize(crf, alpha_, alpha0)

    # compute initialization time
    dgn.time_init = mytime() - dgn.etime 

    # initialize variance parameter
    sigma = 1.0

    # burnin
    for s in range(1, burnin)

        # gibbs sampler
        gibbs_step(crf, alpha_, alpha0)

        # resample alpha
        if prior == true
            if augmentation == true
                alpha_ = resample_alpha_augmentation(alpha_, vec(sum(crf.counts, dims = 2)), sum(crf.r), alpha, alpha0)
            else
                alpha_, accept = resample_alpha(alpha_, vec(sum(crf.counts, dims = 2)), sum(crf.r), alpha, alpha0, sigma)
                sigma *= acceptance_tuning(accept, s)
            end
        end
        
    end

    # compute burnin time
    dgn.time_burnin = mytime() - dgn.etime

    # initialize output
    probs = zeros(k, d, num_samples)
    probs_hdp = zeros((L + (L == 0 ? 1 : 0)), d, num_samples)

    # initialize acceptance rate
    accept = 0.0

    # posterior sampling
    for s in range(1, num_samples)

        # thinning
        for _ in range(1, thin)

            # gibbs sampler
            gibbs_step(crf, alpha_, alpha0)

            # resample alpha
            if prior == true
                if augmentation == true
                    alpha_ = resample_alpha_augmentation(alpha_, vec(sum(crf.counts, dims = 2)), sum(crf.r), alpha, alpha0)
                else
                    alpha_, accept = resample_alpha(alpha_, vec(sum(crf.counts, dims = 2)), sum(crf.r), alpha, alpha0, sigma)
                end
            end
        end

        # update acceptance rates
        if prior == true dgn.accept_alpha += accept end

        # sample baseprobs
        baseprobs = sample_baseprobs(vec(sum(crf.r, dims = 1)), alpha0)

        # sample from stick-breaking
        baseprobs_dp = stick_breaking(alpha0, L)

        # sample probs
        probs[:,:,s], probs_hdp[:,:,s] = sample_probs(crf.counts, baseprobs[begin:end-1], baseprobs[end] * baseprobs_dp, alpha_)

    end

    # rescale acceptance rates
    if prior == true dgn.accept_alpha /= num_samples else dgn.accept_alpha = NaN end
    
    # compute execution time
    dgn.etime = mytime() - dgn.etime

    return probs, probs_hdp, dgn

end # sample_posterior_crf

"""
    sample_posterior_collapsed(counts, alpha0, alpha, num_samples, burnin, thin, L, prior, augmentation)

"""
function sample_posterior_collapsed(counts::Matrix{Int64}, alpha0::Float64, alpha::Float64, num_samples::Int64, burnin::Int64, thin::Int64, L::Int64, prior::Bool, augmentation::Bool)

    # retrieve dimensions
    d, k = size(counts)

    # initialize diagnostics
    dgn = DiagnosticsHDP()

    # compute coefficients
    stirling_coeffs, mincounts = compute_stirling_coeffs(counts)

    # compute initialization time
    dgn.time_init = mytime() - dgn.etime 

    # initialize alpha_
    alpha_ = prior ? (alpha * alpha0) : alpha

    # initialize variance parameter
    sigma = 1.0

    # initialize tables
    tables = ones(Int64, k)

    # burnin
    for s in range(1, burnin)

        # sample number of tables
        for (j, coeffsj) in enumerate(stirling_coeffs)
            tables[j] = sample_tables(sum(tables) - tables[j], mincounts[j], coeffsj, alpha_, alpha0)
        end

        # resample alpha
        if prior == true
            if augmentation == true
                alpha_ = resample_alpha_augmentation(alpha_, vec(sum(counts, dims = 2)), sum(tables), alpha, alpha0)
            else
                alpha_, accept = resample_alpha(alpha_, vec(sum(counts, dims = 2)), sum(tables), alpha, alpha0, sigma)
                sigma *= acceptance_tuning(accept, s)
            end
        end

    end

    # compute burnin time
    dgn.time_burnin = mytime() - dgn.etime

    # initialize output
    probs = zeros(k, d, num_samples)
    probs_hdp = zeros((L + (L == 0 ? 1 : 0)), d, num_samples)

    # initialize acceptance rate
    accept = 0.0

    # posterior sampling
    for s in range(1, num_samples)

        # thinning
        for _ in range(1, thin)

            # sample number of tables
            for (j, coeffsj) in enumerate(stirling_coeffs)
                tables[j] = sample_tables(sum(tables) - tables[j], mincounts[j], coeffsj, alpha_, alpha0)
            end

            # resample alpha
            if prior == true
                if augmentation == true
                    alpha_ = resample_alpha_augmentation(alpha_, vec(sum(counts, dims = 2)), sum(tables), alpha, alpha0)
                else
                    alpha_, accept = resample_alpha(alpha_, vec(sum(counts, dims = 2)), sum(tables), alpha, alpha0, sigma)
                end
            end

        end

        # update acceptance rates
        if prior == true dgn.accept_alpha += accept end

        # sample baseprobs
        baseprobs = sample_baseprobs(tables, alpha0)

        # sample from stick-breaking
        baseprobs_dp = stick_breaking(alpha0, L)

        # sample probs
        probs[:,:,s], probs_hdp[:,:,s] = sample_probs(counts, baseprobs[1:end-1], baseprobs[end] * baseprobs_dp, alpha_)

    end

    # rescale acceptance rates
    if prior == true dgn.accept_alpha /= num_samples else dgn.accept_alpha = NaN end

    # compute execution time
    dgn.etime = mytime() - dgn.etime

    return probs, probs_hdp, dgn

end # sample_posterior_collapsed

"""
    sample_tables(othertables, mincountsj, coeffsj, alpha, alpha0)

"""
function sample_tables(othertables::Int64, mincountsj::Int64, coeffsj::Vector{Float64}, alpha::Float64, alpha0::Float64)

    # initialize masses
    masses = copy(coeffsj)

    # compute probability weights
    cumprod = 1.0
    for h in range(mincountsj, length(masses))
        masses[h] *= cumprod
        cumprod *= alpha / (alpha0 + othertables + h)
    end

    # sample number of tables
    return sample_categorical(masses)

end # sample_tables
