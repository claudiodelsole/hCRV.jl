# load module
using hCRV

# imports
import Statistics: mean, std
import MCMCDiagnosticTools: ess
import Distributions: Gamma, Poisson
import Random: seed!

# sampling models
include("../aux_code/models.jl")

# model parameters
alpha, b = 1.0, 1.0
alpha0, b0 = 1.0, 1.0

# number of samples
num_samples = 10000
burnin = 1000

# number of experiments
num_experiments = 1000

# effective sample sizes and acceptance rates
ess_gamma_latent, accept_gamma_latent = zeros(num_experiments), zeros(num_experiments)
ess_lognormal_latent, accept_lognormal_latent = zeros(num_experiments), zeros(num_experiments)
ess_gamma_jumps, accept_gamma_jumps = zeros(num_experiments), zeros(num_experiments)
ess_lognormal_jumps, accept_lognormal_jumps = zeros(num_experiments), zeros(num_experiments)

# experiments
for s in range(1, num_experiments)

    # set seed
    seed!(110590 + s)

    # number of observations
    d = rand(Poisson(5.0))
    while d == 0 d = rand(Poisson(5.0)) end
    counts_group = fill(100, d)

    # sample observations
    X = model_hdp(counts_group, rand(Gamma(5.0)), rand(Gamma(3.0)))

    # extract info from data
    counts, Xstar = hCRV.setup_hCRV(X)
    d, k = size(counts)

    # effective jumps indexes
    idxs = Vector{Int64}(undef, 0)
    for j in range(1, k)
        if maximum(counts[:,j]) > 1 append!(idxs, j) end
    end

    # ----------
    # Metropolis-Hastings with gamma proposal

    # initialize state
    baselatents, basejumps = ones(k+1), ones(k)

    # initialize variance parameters
    varpar_latent, varpars_jumps = 1.0, ones(k)
    
    # burnin
    for t in range(1, burnin)
        
        # sample baselatent
        accept = hCRV.sample_baselatents_mcmc(baselatents, counts, b0 / alpha, alpha0, false, varpar_latent)
        varpar_latent *= hCRV.acceptance_tuning(accept, t)
        
        # sample latent variables
        latents = hCRV.sample_latents(counts_group, sum(baselatents))
        
        # precompute rate
        rate = b0 / alpha + sum(log.(1.0 .+ latents))
        
        # sample basejumps
        for j in idxs
            basejumps[j], accept = hCRV.sample_basejump_mcmc(basejumps[j], counts[:,j], rate, false, varpars_jumps[j])
            varpars_jumps[j] *= hCRV.acceptance_tuning(accept, t)
        end

    end

    # initialize output
    sumbaselatents, basejumpsout = zeros(num_samples), zeros(k, num_samples)
    acceptance_latent, acceptance_jumps = 0.0, zeros(k)

    # posterior sampling
    for t in range(1, num_samples)

        # sample baselatent
        accept = hCRV.sample_baselatents_mcmc(baselatents, counts, b0 / alpha, alpha0, false, varpar_latent)
        sumbaselatents[t] = sum(baselatents)
        acceptance_latent += accept
        
        # sample latent variables
        latents = hCRV.sample_latents(counts_group, sum(baselatents))
        
        # precompute rate
        rate = b0 / alpha + sum(log.(1.0 .+ latents))
        
        # sample basejumps
        for j in idxs
            basejumps[j], accept = hCRV.sample_basejump_mcmc(basejumps[j], counts[:,j], rate, false, varpars_jumps[j])
            basejumpsout[j,t] = basejumps[j]
            acceptance_jumps[j] += accept
        end
        
    end

    # record effective sample size
    ess_gamma_latent[s] = ess(sumbaselatents)
    ess_gamma_jumps[s] = mean(mapslices(ess, basejumpsout[idxs,:]; dims = 2))

    # record acceptances
    accept_gamma_latent[s] = acceptance_latent / num_samples
    accept_gamma_jumps[s] = mean(acceptance_jumps[idxs]) / num_samples

    # ----------
    # Metropolis-Hastings with lognormal proposal

    # initialize state
    baselatents, basejumps = ones(k+1), ones(k)

    # initialize variance parameters
    varpar_latent, varpars_jumps = 1.0, ones(k)
    
    # burnin
    for t in range(1, burnin)
        
        # sample baselatent
        accept = hCRV.sample_baselatents_mcmc(baselatents, counts, b0 / alpha, alpha0, true, varpar_latent)
        varpar_latent *= hCRV.acceptance_tuning(accept, t)
        
        # sample latent variables
        latents = hCRV.sample_latents(counts_group, sum(baselatents))
        
        # precompute rate
        rate = b0 / alpha + sum(log.(1.0 .+ latents))
        
        # sample basejumps
        for j in idxs
            basejumps[j], accept = hCRV.sample_basejump_mcmc(basejumps[j], counts[:,j], rate, true, varpars_jumps[j])
            varpars_jumps[j] *= hCRV.acceptance_tuning(accept, t)
        end

    end

    # initialize output
    sumbaselatents, basejumpsout = zeros(num_samples), zeros(k, num_samples)
    acceptance_latent, acceptance_jumps = 0.0, zeros(k)

    # posterior sampling
    for t in range(1, num_samples)

        # sample baselatent
        accept = hCRV.sample_baselatents_mcmc(baselatents, counts, b0 / alpha, alpha0, true, varpar_latent)
        sumbaselatents[t] = sum(baselatents)
        acceptance_latent += accept
        
        # sample latent variables
        latents = hCRV.sample_latents(counts_group, sum(baselatents))
        
        # precompute rate
        rate = b0 / alpha + sum(log.(1.0 .+ latents))
        
        # sample basejumps
        for j in idxs
            basejumps[j], accept = hCRV.sample_basejump_mcmc(basejumps[j], counts[:,j], rate, true, varpars_jumps[j])
            basejumpsout[j,t] = basejumps[j]
            acceptance_jumps[j] += accept
        end
        
    end

    # record effective sample size
    ess_lognormal_latent[s] = ess(sumbaselatents)
    ess_lognormal_jumps[s] = mean(mapslices(ess, basejumpsout[idxs,:]; dims = 2))

    # record acceptances
    accept_lognormal_latent[s] = acceptance_latent / num_samples
    accept_lognormal_jumps[s] = mean(acceptance_jumps[idxs]) / num_samples

    print("|")

end

# ----------
# Print results

# print function
print_results(values) = string(mean(values), " (", std(values), ")")

begin
    println(); println()
    println("Latents:")
    println("\t\tESS\t\t\t\t\tacceptance")
    println("MHgamma:\t", print_results(ess_gamma_latent), "\t", print_results(accept_gamma_latent))
    println("MHnormal:\t", print_results(ess_lognormal_latent), "\t", print_results(accept_lognormal_latent))
end

begin
    println()
    println("Jumps:")
    println("\t\tESS\t\t\t\t\tacceptance")
    println("MHgamma:\t", print_results(ess_gamma_jumps), "\t", print_results(accept_gamma_jumps))
    println("MHnormal:\t", print_results(ess_lognormal_jumps), "\t", print_results(accept_lognormal_jumps))
end
