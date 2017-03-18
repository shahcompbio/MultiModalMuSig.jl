module MultiModalMuSig

using Distributions
using NLopt
using Clustering
using StatsFuns

export IMMCTM, MMCTM, cMMCTM, fit!

include("IMMCTM.jl")
include("MMCTM.jl")
include("cMMCTM.jl")
include("common.jl")

end # module
