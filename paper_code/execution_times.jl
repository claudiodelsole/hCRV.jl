# load modules
using hCRV
include("../src/HDP.jl")
using .HDP

# imports
import NPZ: npzwrite
import Statistics: mean
import Distributions: Gamma
import MCMCDiagnosticTools: ess

# sampling models
include("../aux_code/models.jl")

# model parameters
alpha, b = 1.0, 1.0
alpha0, b0 = 1.0, 1.0

# number of samples
num_samples = 2000
burnin = 1000

# average effective sample size
avg_ess(probs) = mean(mapslices(ess, probs; dims = 3))

# number of experiments
num_experiments = 100

##########

# number of groups
dvalues = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

# number of clusters
num_clusters = zeros(Int64, length(dvalues), num_experiments)

# sampling speed and effective samples
time_mcmc, ess_mcmc = zeros(length(dvalues), num_experiments), zeros(length(dvalues), num_experiments)
time_exact, ess_exact = zeros(length(dvalues), num_experiments), zeros(length(dvalues), num_experiments)
time_crf, ess_crf = zeros(length(dvalues), num_experiments), zeros(length(dvalues), num_experiments)
time_collapsed, ess_collapsed = zeros(length(dvalues), num_experiments), zeros(length(dvalues), num_experiments)

for (id, d) in enumerate(dvalues)

    # parameters
    counts_group = fill(25, d)

    for s in range(1, num_experiments)

        # sample observations
        X = model_hdp(counts_group, 3.0, 5.0, seed = 110590 + s)
        
        # Metropolis-Hastings with lognormal proposal
        probs, _, _, Xstar, dgn = posterior_gamma_mcmc(X, alpha0, b0 / alpha, b, num_samples, burnin = burnin, normalize = true, logscale = true)
        num_clusters[id,s] = length(Xstar)
        time_mcmc[id,s], ess_mcmc[id,s] = dgn.etime - dgn.time_burnin, avg_ess(probs)

        # exact sampler
        probs, _, _, _, dgn = posterior_gamma_exact(X, alpha0, b0 / alpha, b, num_samples, normalize = true)
        time_exact[id,s], ess_exact[id,s] = isnothing(dgn) ? (0.0, -1.0) : (dgn.etime - dgn.time_init, avg_ess(probs))

        # restaurant franchise sampler
        probs, _, _, _, dgn = posterior_hdp(X, alpha0, alpha, num_samples, burnin = burnin, prior = true)
        time_crf[id,s], ess_crf[id,s] = dgn.etime - dgn.time_burnin, avg_ess(probs)

        # collapsed Gibbs sampler
        probs, _, _, _, dgn = posterior_hdp(X, alpha0, alpha, num_samples, burnin = burnin, prior = true, collapsed = true)
        time_collapsed[id,s], ess_collapsed[id,s] = isnothing(dgn) ? (0.0, -1.0) : (dgn.etime - dgn.time_burnin, avg_ess(probs))

        print("|")

    end

    println("\tFinished for d = ", d)

end

# number of failures
println("exact:\t", sum(time_exact .== 0.0, dims = 2))
println("collapsed:\t", sum(time_collapsed .== 0.0, dims = 2))

# save times
npzwrite("saves/times_d.npz", 
         Dict("dvalues" => dvalues, "num_clusters" => num_clusters, 
              "mcmc" => time_mcmc, "mcmc_ess" => ess_mcmc,
              "exact" => time_exact, "exact_ess" => ess_exact,
              "crf" => time_crf, "crf_ess" => ess_crf,
              "collapsed" => time_collapsed, "collapsed_ess" => ess_collapsed))



##########

# number of observations per group
nvalues = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

# number of clusters
num_clusters = zeros(Int64, length(nvalues), num_experiments)

# sampling speed and effective samples
time_mcmc, ess_mcmc = zeros(length(nvalues), num_experiments), zeros(length(nvalues), num_experiments)
time_exact, ess_exact = zeros(length(nvalues), num_experiments), zeros(length(nvalues), num_experiments)
time_crf, ess_crf = zeros(length(nvalues), num_experiments), zeros(length(nvalues), num_experiments)
time_collapsed, ess_collapsed = zeros(length(nvalues), num_experiments), zeros(length(nvalues), num_experiments)

for (id, n) in enumerate(nvalues)

    # parameters
    counts_group = fill(n, 10)

    for s in range(1, num_experiments)

        # sample observations
        X = model_hdp(counts_group, 3.0, 5.0, seed = 110590 + s)
        
        # Metropolis-Hastings with lognormal proposal
        probs, _, _, Xstar, dgn = posterior_gamma_mcmc(X, alpha0, b0 / alpha, b, num_samples, burnin = burnin, normalize = true, logscale = true)
        num_clusters[id,s] = length(Xstar)
        time_mcmc[id,s], ess_mcmc[id,s] = dgn.etime - dgn.time_burnin, avg_ess(probs)

        # exact sampler
        probs, _, _, _, dgn = posterior_gamma_exact(X, alpha0, b0 / alpha, b, num_samples, normalize = true)
        time_exact[id,s], ess_exact[id,s] = isnothing(dgn) ? (0.0, -1.0) : (dgn.etime - dgn.time_init, avg_ess(probs))

        # restaurant franchise sampler
        probs, _, _, _, dgn = posterior_hdp(X, alpha0, alpha, num_samples, burnin = burnin, prior = true)
        time_crf[id,s], ess_crf[id,s] = dgn.etime - dgn.time_burnin, avg_ess(probs)

        # collapsed Gibbs sampler
        probs, _, _, _, dgn = posterior_hdp(X, alpha0, alpha, num_samples, burnin = burnin, prior = true, collapsed = true)
        time_collapsed[id,s], ess_collapsed[id,s] = isnothing(dgn) ? (0.0, -1.0) : (dgn.etime - dgn.time_burnin, avg_ess(probs))

        print("|")

    end

    println("\tFinished for n = " , n)

end

# number of failures
println("exact:\t", sum(time_exact .== 0.0, dims = 2))
println("collapsed:\t", sum(time_collapsed .== 0.0, dims = 2))

# save times
npzwrite("saves/times_n.npz", 
         Dict("nvalues" => nvalues, "num_clusters" => num_clusters, 
              "mcmc" => time_mcmc, "mcmc_ess" => ess_mcmc,
              "exact" => time_exact, "exact_ess" => ess_exact,
              "crf" => time_crf, "crf_ess" => ess_crf,
              "collapsed" => time_collapsed, "collapsed_ess" => ess_collapsed))

##########

# number of clusters
kvalues = [4, 5, 6, 7, 8, 9, 10, 11, 12]

# number of experiments per cluster
counter = zeros(Int64, length(kvalues))

# sampling speed and effective samples
time_mcmc, ess_mcmc = zeros(length(kvalues), num_experiments), zeros(length(kvalues), num_experiments)
time_exact, ess_exact = zeros(length(kvalues), num_experiments), zeros(length(kvalues), num_experiments)
time_crf, ess_crf = zeros(length(kvalues), num_experiments), zeros(length(kvalues), num_experiments)
time_collapsed, ess_collapsed = zeros(length(kvalues), num_experiments), zeros(length(kvalues), num_experiments)

# parameters
counts_group = fill(25, 10)

# set seed
myseed = 180396

while sum(counter .< num_experiments) > 0

    # sample observations
    X = model_hdp(counts_group, rand(Gamma(3.0)), rand(Gamma(5.0)), seed = (myseed += 1))

    # number of clusters
    _, Xstar = hCRV.setup_hcrv(X)
    id = findfirst(kvalues .== length(Xstar))
    if isnothing(id) continue end

    # update counter
    s = (counter[id] += 1)
    if s > num_experiments continue end

    # Metropolis-Hastings with lognormal proposal
    probs, _, _, _, dgn = posterior_gamma_mcmc(X, alpha0, b0 / alpha, b, num_samples, burnin = burnin, normalize = true, logscale = true)
    time_mcmc[id,s], ess_mcmc[id,s] = dgn.etime - dgn.time_burnin, avg_ess(probs)

    # exact sampler
    probs, _, _, _, dgn = posterior_gamma_exact(X, alpha0, b0 / alpha, b, num_samples, normalize = true)
    time_exact[id,s], ess_exact[id,s] = isnothing(dgn) ? (0.0, -1.0) : (dgn.etime - dgn.time_init, avg_ess(probs))

    # restaurant franchise sampler
    probs, _, _, _, dgn = posterior_hdp(X, alpha0, alpha, num_samples, burnin = burnin, prior = true)
    time_crf[id,s], ess_crf[id,s] = dgn.etime - dgn.time_burnin, avg_ess(probs)

    # collapsed Gibbs sampler
    probs, _, _, _, dgn = posterior_hdp(X, alpha0, alpha, num_samples, burnin = burnin, prior = true, collapsed = true)
    time_collapsed[id,s], ess_collapsed[id,s] = isnothing(dgn) ? (0.0, -1.0) : (dgn.etime - dgn.time_init, avg_ess(probs))

    print("|")

end

# number of failures
println("exact:\t", sum(time_exact .== 0.0, dims = 2))
println("collapsed:\t", sum(time_collapsed .== 0.0, dims = 2))

# save times
npzwrite("saves/times_k.npz", 
         Dict("kvalues" => kvalues,
              "mcmc" => time_mcmc, "mcmc_ess" => ess_mcmc,
              "exact" => time_exact, "exact_ess" => ess_exact,
              "crf" => time_crf, "crf_ess" => ess_crf,
              "collapsed" => time_collapsed, "collapsed_ess" => ess_collapsed))
