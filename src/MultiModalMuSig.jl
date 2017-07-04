module MultiModalMuSig

using Distributions
using NLopt
using Clustering
using StatsFuns

export IMMCTM, MMCTM, LDA, fit!

include("IMMCTM.jl")
include("MMCTM.jl")
include("LDA.jl")
include("common.jl")

end # module
