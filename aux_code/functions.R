# activate Julia project
julia_command('using Pkg')
julia_command('Pkg.activate(".")')
julia_command('Pkg.resolve()')
julia_command('Pkg.update()')
julia_command('Pkg.instantiate()')

# load packages
julia_command("using hCRV")
julia_call("include", "src/HDP.jl")
julia_command("using .HDP")

# create julia data set
julia_dataset <- function(X) {
  
  # initialize output
  julia_eval("X = Vector{Vector{Int64}}(undef, 0)")

  # loop over groups
  for (X_group in X) {
    julia_assign("X_group", X_group)  # assign in Julia
    julia_eval("push!(X, X_group)") # append to vector
  }

  julia_eval("X")
}

# posterior_gamma_mcmc
posterior_gamma_mcmc <- function(X, alpha0, rate0, b, num_samples, 
                                 burnin = 0, thin = 1, L = 0, normalize = F, logscale = T) {
  
  # create Julia data set
  localX = julia_dataset(X)

  # call Julia function
  out <- julia_call("posterior_gamma_mcmc", localX, alpha0, rate0, b, as.integer(num_samples), 
                    burnin = as.integer(burnin), thin = as.integer(thin), L = as.integer(L), 
                    normalize = normalize, logscale = logscale)

  list(jumps = out[[1]], jumpsc = out[[2]], 
       counts = out[[3]], Xstar = out[[4]], dgn = diagnostics_mcmc(out[[5]]))
}

# diagnostics_mcmc
diagnostics_mcmc <- function(dgn) {
  
  # extract from Julia object
  julia_assign("dgn", dgn)
  list(accept_latent = julia_eval("dgn.accept_latent"),
       accept_jumps = julia_eval("dgn.accept_jumps"),
       time_burnin = julia_eval("dgn.time_burnin"),
       etime = julia_eval("dgn.etime"))
}

# posterior_gamma_exact
posterior_gamma_exact <- function(X, alpha0, rate0, b, num_samples, 
                                 L = 0, normalize = F) {
  
  # create Julia data set
  localX = julia_dataset(X)

  # call Julia function
  out <- julia_call("posterior_gamma_exact", localX, alpha0, rate0, b, as.integer(num_samples), 
                    L = as.integer(L), normalize = normalize)

  list(jumps = out[[1]], jumpsc = out[[2]], 
       counts = out[[3]], Xstar = out[[4]], dgn = diagnostics_exact(out[[5]]))
}

# diagnostics_exact
diagnostics_exact <- function(dgn) {
  
  # extract from Julia object
  julia_assign("dgn", dgn)
  list(accept_latent = julia_eval("dgn.accept_latent"),
       time_init = julia_eval("dgn.time_init"),
       time_latent = julia_eval("dgn.time_latent"),
       etime = julia_eval("dgn.etime"))
}

# posterior_hdp
posterior_hdp <- function(X, alpha0, alpha, num_samples, 
                          burnin = 0, thin = 1, L = 0, prior = F, collapsed = F) {
  
  # create Julia data set
  localX = julia_dataset(X)
  
  # call Julia function
  out <- julia_call("posterior_hdp", localX, alpha0, alpha, as.integer(num_samples), 
                    burnin = as.integer(burnin), thin = as.integer(thin), L = as.integer(L),
                    prior = prior, collapsed = collapsed)
  
  list(jumps = out[[1]], jumpsc = out[[2]], 
       counts = out[[3]], Xstar = out[[4]], dgn = diagnostics_hdp(out[[5]]))
}

# diagnostics_hdp
diagnostics_hdp <- function(dgn) {

  # extract from Julia object
  julia_assign("dgn", dgn)
  list(accept_alpha = julia_eval("dgn.accept_alpha"),
       time_init = julia_eval("dgn.time_init"),
       time_burnin = julia_eval("dgn.time_burnin"),
       etime = julia_eval("dgn.etime"))
}
