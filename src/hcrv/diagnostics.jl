# export structs
export DiagnosticsMCMC, DiagnosticsExact

"""
    mutable struct DiagnosticsMCMC

Store diagnostics for posterior sampling in [`posterior_gamma_mcmc`](@ref).

# Fields
- `accept_latent`: acceptance rate for latent variable updates
- `accept_jumps`: acceptance rates for each basejump
- `time_burnin`: execution time for the burnin phase
- `etime`: total execution time
"""
mutable struct DiagnosticsMCMC

    # acceptance rates
    accept_latent::Float64
    accept_jumps::Vector{Float64}

    # execution times
    time_burnin::Float64
    etime::Float64

    # constructor
    DiagnosticsMCMC(k::Int) = new(0.0, zeros(k), 0.0, mytime())
    
end # struct

"""
   mutable struct DiagnosticsExact 

Store diagnostics for posterior sampling in [`posterior_gamma_exact`](@ref).

# Fields
- `accept_latent`: acceptance rate for latent variable rejection sampler
- `time_init`: execution time for initialization
- `time_latent`: execution time for latent variable rejection sampler
- `etime`: total execution time
"""
mutable struct DiagnosticsExact

    # acceptance rates
    accept_latent::Float64

    # execution times
    time_init::Float64
    time_latent::Float64
    etime::Float64
    
    # constructor
    DiagnosticsExact() = new(0.0, 0.0, 0.0, mytime())
    
end # struct
