module MultiModalMuSig

using Distributions
using NLopt
using Clustering
using StatsFuns

export IMMCTM, MMCTM, ILDA, LDA, fit!

include("IMMCTM.jl")
include("MMCTM.jl")
include("ILDA.jl")
include("LDA.jl")
include("common.jl")

end # module
