"""
   module HDP

"""
module HDP

# imports
include("imports.jl")

# utilities
include("utils.jl")
include("hdp/stirling_coeffs.jl")

# posterior sampling
include("hdp/crf.jl")
include("hdp/posterior.jl")
include("hdp/probs.jl")

# resampling concentration
include("hdp/resampling.jl")

# diagnostics
include("hdp/diagnostics.jl")

end # module HDP
