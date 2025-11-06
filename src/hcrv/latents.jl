"""
    sample_baselatents_mcmc(latents, counts, rate, alpha0, logscale, varpar)

"""
function sample_baselatents_mcmc(latents::Vector{Float64}, counts::Matrix{Int64}, rate::Float64, alpha0::Float64, logscale::Bool, varpar::Float64)::Float64

    # retrieve dimension
    k = size(counts, 2)
    
    # loop on components
    # for _ in range(1, div(k, 2))
    for _ in range(1, k)

        # choose components
        j, l = rand(1:(k+1)), rand(1:(k+1))
        while l == j l = rand(1:(k+1)) end
        
        # order components
        if j > l j, l = l, j end
        
        # retrieve counts
        countsj = counts[:,j]
        countsl = l <= k ? counts[:,l] : zeros(Int, 1)

        # retrieve latents
        tj, tl = latents[j], latents[l]
        sumt = tj + tl
        
        # special Dirichlet case
        if maximum(countsj) == 1 && maximum(countsl) == 1
            probs = rand(Dirichlet([sum(countsj), sum(countsl)]))
            latents[j], latents[l] = sumt * probs[1], sumt * probs[2]
            continue
        end
        
        # special Dirichlet case
        if maximum(countsj) == 1 && sum(countsl) == 0
            probs = rand(Dirichlet([sum(countsj), alpha0]))
            latents[j], latents[l] = sumt * probs[1], sumt * probs[2]
            continue
        end
        
        # propose values
        unif = rand()
        tj_new, tl_new = (1.0 - unif) * sumt, unif * sumt
        
        # compute acceptance probability
        logaccept = loglikelihood_simplex(tj_new, countsj, alpha0) + loglikelihood_simplex(tl_new, countsl, alpha0)
        logaccept -= loglikelihood_simplex(tj, countsj, alpha0) + loglikelihood_simplex(tl, countsl, alpha0)
        
        # accept / reject
        if log(rand()) < logaccept
            latents[j], latents[l] = tj_new, tl_new
        end

    end
    
    # compute delta
    delta = 1.0 / varpar
    
    # propose perturbation
    eps = logscale ? exp(sqrt(varpar) * randn()) : rand(Gamma(delta)) / delta
    
    # compute acceptance probability
    logaccept = loglikelihood_baselatents(latents .* eps, counts, rate, alpha0)
    logaccept -= loglikelihood_baselatents(latents, counts, rate, alpha0)
    if !logscale logaccept += delta * (eps - 1.0/eps) - 2.0 * delta * log(eps) end
    
    # accept / reject
    if log(rand()) < logaccept
        latents .*= eps
    end
    
    return min(exp(logaccept), 1.0)

end # sample_baselatents_mcmc

"""
    loglikelihood_baselatents(t, counts, rate, alpha0)

"""
function loglikelihood_baselatents(t::Vector{Float64}, counts::Matrix{Int64}, rate::Float64, alpha0::Float64)::Float64
    
    # compute sum
    sumt = sum(t)
    
    # compute loglikelihood
    logpdf = alpha0 * log(t[end]) - rate * sumt
    
    # rising factorials
    for (tj, countsj) in zip(t[begin:(end-1)], eachcol(counts))
        for count in countsj
            logpdf += rising_factorial(tj, count)
        end
    end
    
    # rising factorials
    for count in sum(counts, dims = 2)
        logpdf -= rising_factorial(sumt, count)
    end
    
    return logpdf

end # loglikelihood_baselatents

"""
    loglikelihood_simplex(tj, countsj, alpha0)

"""
function loglikelihood_simplex(tj::Float64, countsj::Vector{Int64}, alpha0::Float64)::Float64
    
    # special case
    if sum(countsj) == 0
        return (alpha0 - 1.0) * log(tj)
    end
    
    # compute logpdf
    logpdf = - log(tj)
    for count in countsj
        logpdf += rising_factorial(tj, count)
    end
    
    return logpdf

end # loglikelihood_simplex

"""
    sample_baselatent_rejection(weights, counts_group, rate, alpha0, num_samples)

"""
function sample_baselatent_rejection(weights::Vector{Float64}, counts_group::Vector{Int64}, rate::Float64, alpha0::Float64, num_samples::Int64)

    # retrieve dimensions
    d, mincounts = length(counts_group), findfirst(x -> x != 0.0, weights)
    
    # create polynomial
    polycoeffs = compute_coefficients(weights, mincounts, alpha0)
    poly = Polynomial(polycoeffs)

    # define function
    opt_shape(r::Float64) = optimal_shape_equation(r, poly, counts_group, mincounts, rate, alpha0)
    
    # find optimal shape parameter
    if opt_shape(0.0) <= 0.0  ropt = 0.0    # extreme case
    elseif opt_shape(Float64(mincounts - d)) >= 0.0 ropt = Float64(mincounts - d)   # extreme case
    else ropt = find_zero(opt_shape, (0.0, Float64(mincounts - d)), Bisection()) end    # optimization
    
    # find upper bound
    ub = - optimize(t::Float64 -> - logaccept_baselatent(exp(t), poly, counts_group, mincounts, ropt), -8.0, 8.0).minimum
    
    # acceptance counters
    accepted, counter = 0, 0
    
    # initialize output
    baselatent_samples = Vector{Float64}(undef, num_samples)
    
    # rejection sampling
    while accepted < num_samples

        # sample proposal from gamma distribution
        baselatent = rand(Gamma(alpha0 + ropt)) / rate
        
        # compute acceptance probability
        logaccept = logaccept_baselatent(baselatent, poly, counts_group, mincounts, ropt)
        
        # check for upper bound
        if logaccept > ub
            @warn "Rejection sampling: invalid upper bound!"
            ub = logaccept
        end
        
        # accept / reject
        if log(rand()) + ub < logaccept
            accepted += 1
            baselatent_samples[accepted] = baselatent
        end
        
        # update counter
        counter += 1

    end
    
    return baselatent_samples, num_samples / counter, ropt, ub

end # sample_baselatent_rejection

"""
    logaccept_baselatent(t::Float64, poly::Polynomial, counts_group::Vector{Int64}, mincounts::Int64, ropt::Float64)::Float64

"""
function logaccept_baselatent(t::Float64, poly::Polynomial, counts_group::Vector{Int64}, mincounts::Int64, ropt::Float64)::Float64

    # limiting cases
    if log(t) >= 8.0
        return mincounts + length(poly.coeffs) == sum(counts_group) + 1 && ropt == 0.0 ? log(poly.coeffs[end]) : - Inf
    end
    
    # compute logaccept
    logaccept = (mincounts - length(counts_group) - ropt) * log(t)
    
    # polynomial term
    logaccept += log(poly(t))
    
    # rising factorials
    for count in counts_group
        logaccept -= rising_factorial(t + 1.0, count - 1)
    end
    
    return logaccept

end # logaccept_baselatent

"""
    optimal_shape_equation(r, poly, counts_group, mincounts, rate, alpha0)

"""
function optimal_shape_equation(r::Float64, poly::Polynomial, counts_group::Vector{Int64}, mincounts::Int, rate::Float64, alpha0::Float64)::Float64

    # compute upper bound
    logtstar = optimize(t::Float64 -> - logaccept_baselatent(exp(t), poly, counts_group, mincounts, r), -8.0, 8.0).minimizer
    
    # compute function value
    value = logtstar + log(rate) - digamma(alpha0 + r)
    
    return value

end # optimal_shape_equation

"""
    logdensity_baselatent(t, poly, counts_group, mincounts, rate, alpha0)

"""
function logdensity_baselatent(t::Float64, poly::Polynomial, counts_group::Vector{Int64}, mincounts::Int64, rate::Float64, alpha0::Float64)::Float64
    
    # compute logdensity
    logpdf = (alpha0 + mincounts - 1.0) * log(t) - rate * t
    
    # polynomial term
    logpdf += log(poly(t))
    
    # rising factorials
    for count in counts_group
        logpdf -= rising_factorial(t, count)
    end
    
    return logpdf

end # logdensity_baselatent

"""
    logdensity_baselatent_derivative(t, poly, counts_group, mincounts, rate, alpha0)

"""
function logdensity_baselatent_derivative(t::Float64, poly::Polynomial, counts_group::Vector{Int64}, mincounts::Int64, rate::Float64, alpha0::Float64)::Float64

    # compute dlogdensity
    logpdf = (alpha0 + mincounts - 1.0) / t - rate
    
    # polynomial derivative term
    poly_derivative = derivative(poly)
    logpdf += poly_derivative(t) / poly(t)
    
    # rising factorials derivatives
    for count in counts_group
        logpdf -= rising_factorial_derivative(t, count)
    end
    
    return logpdf

end # logdensity_baselatent_derivative

"""
    compute_coefficients(weights, alpha0)

"""
function compute_coefficients(weights::Vector{Float64}, mincounts::Int64, alpha0::Float64)::Vector{Float64}
    
    # initialize weights
    weights_scaled = copy(weights)
    
    # rescale weights
    cumprod = 1.0
    for h in range(mincounts, length(weights))
        weights_scaled[h] /= cumprod
        cumprod *= (alpha0 + h)
    end
    
    # filter positive weights
    idxs = weights_scaled .> 0
    return weights_scaled[idxs]

end # compute_coefficients

"""
    sample_latents(counts_group, baselatent)

"""
function sample_latents(counts_group::Vector{Int64}, baselatent::Float64)::Vector{Float64}
    
    # initialize output
    latents = Vector{Float64}(undef, length(counts_group))
    
    # sample from gamma distributions
    for (i, count) in enumerate(counts_group)
        beta = baselatent > 0.01 ? rand(Gamma(baselatent)) : baselatent
        latents[i] = rand(Gamma(count)) / beta
    end
    
    return latents

end # sample_latents
