# Hierarchical completely random vectors

This repository contains the source code and scripts to replicate results in the working paper [*Hierarchical Random Measures without Tables*](https://arxiv.org/abs/2505.02653) by Marta Catalano and Claudio Del Sole. The core source code is organized in the form of [Julia](https://julialang.org/) package; an interface to allow integration within the R environment is also provided.

The source code is available in the `/src` folder, and implements the MCMC and exact posterior sampling algorithms for the gamma-gamma hierarchical CRV model introduced in the paper (Section 5.3). An implementation of the CRF-based and collapsed Gibbs samplers for the hierarchical Dirichlet process is also available (Sections 5.3 and S3.7).

The folder `/paper_code` collects the scripts to reproduce the analyses in the paper (numbers, figures and tables).

## Installation and usage in Julia

This repository is in the form of Julia package. To install and load the package:
* clone this repository in a local folder;
* open a Julia REPL (i.e. the interactive command-line interface for Julia) inside that folder;
* create a Julia environment by running `using Pkg` and `Pkg.activate(".")`: the local environment is named `hCRV` and contains the information in `Project.toml`;
* download and install the required dependencies by running `Pkg.instantiate()`: the exact version of each dependency is listed in `Manifest.toml`;
* load the package by running `using hCRV`.

The package exports the main functions for posterior sampling, namely `posterior_gamma_mcmc` and `posterior_gamma_exact`. Details on their arguments and outputs are available in their documentation, which can be accessed by running `?(function_name)`, for example `?posterior_gamma_mcmc`.
Their outputs also include structs with diagnostics of the sampling algorithms, named `DiagnosticsMCMC` and `DiagnosticsExact`; details of their fields can be obtained by accessing their documentation. 

The additional module `HDP` export the function `posterior_hdp`, implementing the posterior sampling algorithms for the hierarchical Dirichlet process, and the struct with diagnostics `DiagnosticsHDP`. The module can be loaded by running `include("src/HDP.jl")` and then `using .HDP`.

Finally, the file `/aux_code/models.jl` contains auxiliary functions to generate synthetic observations; for example, `model_poisson` generates indpendent observations from Poisson distributions.

A complete minimum working example is the following:

```
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
```

## R interface

The script `hcrv.R` contains instructions to integrate the Julia package within the R environment. The installation and setup of the required Julia dependencies is managed by the script `/aux_code/interface.R`, which also provides the interface to run Julia functions in R. Make sure to have Julia installed before executing the instructions in the `hcrv.R` script.
