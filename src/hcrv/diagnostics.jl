# export structs
export DiagnosticsMCMC, DiagnosticsExact

"""
    mutable struct DiagnosticsMCMC

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

# """
#     mutable struct DiagnosticsMixtures 

# """
# mutable struct DiagnosticsMixtures

#     # execution times
#     time_burnin::Float64
#     etime::Float64
    
#     # constructor
#     DiagnosticsMixtures() = new(0.0, mytime())

# end # struct
