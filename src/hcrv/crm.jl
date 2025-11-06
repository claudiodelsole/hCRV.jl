"""
    gamma_process(alpha, rate, L)

"""
function gamma_process(alpha::Float64, rate::Float64, L::Int64)::Vector{Float64}
    
    # total mass
    if L == 0 return [rand(Gamma(alpha)) / rate] end
    
    # sample Poisson process
    poisson = cumsum(rand(Exponential(), L)) / alpha
    
    # initialize jumps
    jump, jumps = 1.0, Vector{Float64}(undef, L)
    
    # invert exponential integral
    for (l, xi) in enumerate(poisson)
        jump = inverse_tail_integral(xi, jump) / rate
        jumps[l] = jump
    end
    
    return jumps

end # gamma_process

"""
    tail_integral(logx)

"""
function tail_integral(logx::Float64)::Float64
    
    if logx <= -256.0   # asymptotic behaviour
        return - eulergamma - logx 
    end
    
    # return expint(exp(logx))
    return gamma(0.0, exp(logx))

end # tail_integral

"""
    tail_integral_derivative(logx)

"""
function tail_integral_derivative(logx::Float64)::Float64

    if logx <= -256.0   # asymptotic behaviour
        return - 1.0
    end
    
    return - exp( - exp(logx))

end # tail_integral_derivative

"""
    inverse_tail_integral(values)

"""
function inverse_tail_integral(value::Float64, pvalue::Float64)::Float64
    
    # solve equation
    logjump = newton(value, tail_integral, tail_integral_derivative, log(pvalue))
    # logjump = find_zero((logx::Float64 -> tail_integral(logx) - value, tail_integral_derivative), log(pvalue), Newton(), atol = 1.0e-8, maxevals = 100)
    
    return exp(logjump)

end # inverse_tail_integral

"""
    sample_jumps_crm(basejumps, latents, b)

"""
function sample_jumps_crm(basejumps::Vector{Float64}, latents::Vector{Float64}, b::Float64)::Matrix{Float64}
    
    # initialize output
    jumps = zeros(length(basejumps), length(latents))
    
    # sample from gamma distribution
    for (l, jump) in enumerate(basejumps)
        if jump > 0.0
            for (i, latent) in enumerate(latents)
                jumps[l,i] = rand(Gamma(jump)) / (b + b * latent)
            end
        end
    end
    
    return jumps

end # sample_jumps_crm
