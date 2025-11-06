# export structs
export DiagnosticsHDP

"""
    mutable struct DiagnosticsHDP

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
