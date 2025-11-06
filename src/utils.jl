"""
    rising_factorial(value, steps)

"""
function rising_factorial(value::Float64, steps::Int64)::Float64
    
    # special case
    if steps == 0 return 0.0 end
    
    # using gamma functions for large steps
    # if steps > 20 
        return loggamma(value + steps) - loggamma(value) 
    # end
    
    # cumulative products
    # rfact = 1.0
    # for h in range(0, steps-1)
    #     rfact *= (value + h)
    # end
    
    # return log(rfact)

end # rising_factorial

"""
    rising_factorial_derivative(value, steps)

"""
function rising_factorial_derivative(value::Float64, steps::Int64)::Float64
    
    # special case
    if steps == 0 return 0.0 end
    
    # using gamma functions for large steps
    # if steps > 20 
        return digamma(value + steps) - digamma(value) 
    # end
    
    # cumulative products
    # logrfact = 0.0
    # for h in range(0, steps-1)
    #     logrfact += 1.0 / (value + h)
    # end
    
    # return logrfact

end # rising_factorial_derivative

"""
    acceptance_tuning(accept, n; n0 = 10, target = 0.44)

"""
function acceptance_tuning(accept::Float64, n::Int64; n0::Int64 = 10, target::Float64 = 0.44)::Float64
    
    # exact sampling
    if isnan(accept) return 1.0 end
    
    return exp((accept - target) / sqrt(n0 + n))

end # acceptance_tuning

"""
    generalized_stirling_numbers(counts)

"""
function generalized_stirling_numbers(counts::Vector{Int64})::Vector{Float64}
    
    # initialize weights
    coeffs = zeros(Float64, sum(counts) + 1)
    coeffs[1], nsum = 1.0, 0
    
    # compute weights
    for count in counts
        for n in range(0, count-1)
            # coeffs[2:(nsum+2)] = n * coeffs[2:(nsum+2)] + coeffs[1:(nsum+1)]
            for h in range(nsum + 1, 1, step = -1)
                coeffs[h+1] = n * coeffs[h+1] + coeffs[h]
            end
            if nsum == 0 coeffs[1] = 0 end
            nsum += 1
        end
    end
    
    return coeffs[2:end]

end # generalized_stirling_numbers

"""
    newton(value, f, fp, start)

"""
function newton(value::Float64, f::Function, fp::Function, start::Float64)::Float64

    # algorithm parameters
    tol = 1.0e-8        # tolerance
    maxIter = 100       # number of iterations

    # initialize
    sol, fsol = start, f(start)

    for _ in range(1, maxIter)

        # algorithm step
        sol -= (fsol - value) / fp(sol)

        # precompute function
        fsol = f(sol)

        # check convergence
        if abs(fsol - value) < tol
            return sol
        end

    end

    # !! algorithm does not converge !!
    return -Inf

end # newton

"""
    sample_categorical(masses)

"""
function sample_categorical(masses::Vector{Float64})::Int64

    # sample from uniform distribution
    th = rand() * sum(masses)

    # initialize variables
    K, sum_masses = 1, masses[begin]

    # sample from categorical distribution
    while sum_masses < th
        K += 1
        sum_masses += masses[K]
    end

    # return sampled category
    return K

end # sample_categorical

"""
    mytime()

"""
mytime()::Float64 = 1.0e-9 * time_ns()
