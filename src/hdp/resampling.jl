"""
    resample_alpha(alpha_, counts_group, tables, alpha, alpha0, sigma)

"""
function resample_alpha(alpha_::Float64, counts_group::Vector{Int64}, tables::Int64, alpha::Float64, alpha0::Float64, sigma::Float64)::Tuple{Float64, Float64}

    # propose perturbation
    eps = exp(sigma * randn())

    # compute acceptance probability
    logaccept = loglikelihood_alpha(alpha_ * eps, counts_group, tables, alpha, alpha0)
    logaccept -= loglikelihood_alpha(alpha_, counts_group, tables, alpha, alpha0)

    # accept / reject
    if log(rand()) < logaccept
        alpha_ *= eps
    end

    return alpha_, min(exp(logaccept), 1.0)

end # resample_alpha

"""
    loglikelihood_alpha(s, counts_group, tables, alpha, alpha0)

"""
function loglikelihood_alpha(s::Float64, counts_group::Vector{Int64}, tables::Int64, alpha::Float64, alpha0::Float64)::Float64

    # compute loglikelihood
    loglik = (alpha0 + tables) * log(s) - s / alpha

    # rising factorials
    for count in counts_group
        loglik -= rising_factorial(s, count)
    end

    return loglik

end # loglikelihood_alpha

"""
    resample_alpha_augmentation(alpha_, counts_group, tables, alpha, alpha0)

"""
function resample_alpha_augmentation(alpha_::Float64, counts_group::Vector{Int64}, tables::Int64, alpha::Float64, alpha0::Float64)::Float64

    # initialize rate
    rate = 1.0 / alpha

    # sample from betas
    for count in counts_group
        rate -= log(rand(Beta(alpha_, count)))
    end

    # resample alpha
    alpha_ = rand(Gamma(alpha0 + tables)) / rate

    return alpha_

end # resample_alpha_augmentation