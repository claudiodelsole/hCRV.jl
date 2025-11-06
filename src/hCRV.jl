"""
    module hCRV

"""
module hCRV

# imports
include("imports.jl")

# utilities
include("utils.jl")
include("hcrv/weights.jl")
# include("ars.jl")

# posterior sampling
include("hcrv/posterior.jl")
include("hcrv/latents.jl")
include("hcrv/jumps.jl")
include("hcrv/crm.jl")

# diagnostics
include("hcrv/diagnostics.jl")

end # module hCRV