module MultiModalMuSig

using Distributions
using NLopt
using Clustering
using StatsFuns

export IMMCTM, MMCTM, fit!

include("IMMCTM.jl")
include("MMCTM.jl")
include("common.jl")

end # module
