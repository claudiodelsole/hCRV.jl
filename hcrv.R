# install JuliaCall package
# install.packages("JuliaCall")

# load package: integrates Julia with R
library(JuliaCall)

# setup Julia for use in R
# takes few minutes the first time, some seconds in following calls
julia_setup()

# if Julia is not found automatically, specify the location of the julia.exe file
# julia_setup(JULIA_HOME = "C:/.../bin")
julia_setup(JULIA_HOME = "C:/Users/claud/.julia/juliaup/julia-1.11.1+0.x64.w64.mingw32/bin")

# set working directory to the location of Project.toml file
# setwd("C:/.../hCRV")
setwd("C:/Users/claud/OneDrive - Università Commerciale Luigi Bocconi/Code/hCRV")

# precompile hCRV package and load R auxiliary functions
# takes up to 30/60 seconds
source("aux_code/interface.R")

# data set in R (list of vectors)
X = list(c(4, 2, 4, 4, 5, 5, 7, 3, 2, 5), 
         c(1, 5, 2, 1, 0, 5, 3, 2, 3, 5), 
         c(2, 2, 1, 2, 1, 1, 2, 2, 3, 3))

# model parameters
alpha <- 1.0; b <- 1.0
alpha0 <- 1.0; b0 <- 1.0

# number of samples
num_samples <- 10000
burnin <- 1000

### posterior sampling

# MCMC sampler
out_mcmc <- posterior_gamma_mcmc(X, alpha0, b0 / alpha, b, num_samples, burnin = burnin)

# exact sampler
out_exact <- posterior_gamma_exact(X, alpha0, b0 / alpha, b, num_samples)

# collapsed Gibbs sampler for HDP with matching prior
out_hdp <- posterior_hdp(X, alpha0, alpha, num_samples, burnin = burnin, prior = T, collapsed = T)
