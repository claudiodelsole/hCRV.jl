# create local environment
using Pkg
Pkg.activate(".")

# download and install dependencies
Pkg.instantiate()

# load packages
using hCRV
include("src/HDP.jl")
using .HDP

# include auxiliary functions
include("aux_code/models.jl")

# dataset parameters
counts_group = fill(50, 4)
means = [2.0, 3.0, 4.0, 5.0]

# generate synthetic observations
X = model_poisson(counts_group, means)

# model parameters
alpha, b = 1.0, 1.0
alpha0, b0 = 1.0, 1.0

# number of samples
num_samples = 10000

# MCMC sampler
probs_mcmc, probsc_mcmc, counts, Xstar, dgn_mcmc = posterior_gamma_mcmc(X, alpha0, b0 / alpha, b, num_samples, burnin = 1000, normalize = true)

# exact sampler
probs_exact, probsc_exact, counts, Xstar, dgn_exact = posterior_gamma_exact(X, alpha0, b0 / alpha, b, num_samples, normalize = true)

# collapsed Gibbs sampler for HDP with matching prior
probs_hdppr, probsc_hdppr, counts, Xstar, dgn_hdppr = posterior_hdp(X, alpha0, alpha, num_samples, burnin = 1000, prior = true, collapsed = true)