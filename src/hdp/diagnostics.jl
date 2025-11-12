# export structs
export DiagnosticsHDP

"""
    mutable struct DiagnosticsHDP

Store diagnostics for posterior sampling in [`posterior_hdp`](@ref).

# Fields
- `accept_alpha`: acceptance rate for concentration parameter (if prior = true)
- `time_init`: execution time for initialization
- `time_burnin`: execution time for the burnin phase
- `etime`: total execution time
"""
mutable struct DiagnosticsHDP

    # acceptance rates
    accept_alpha::Float64

    # execution times
    time_init::Float64
    time_burnin::Float64
    etime::Float64

    # constructor
    DiagnosticsHDP() = new(0.0, 0.0, 0.0, mytime())
    
end # struct
