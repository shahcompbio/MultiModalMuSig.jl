module MultiModalMuSig

using LinearAlgebra
using Statistics
using SpecialFunctions
using NLopt

export IMMCTM, MMCTM, ILDA, LDA, fit!

include("IMMCTM.jl")
include("MMCTM.jl")
include("ILDA.jl")
include("LDA.jl")
include("common.jl")

end # module
