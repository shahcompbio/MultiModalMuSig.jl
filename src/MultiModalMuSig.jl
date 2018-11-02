module MultiModalMuSig

using DataFrames
using LinearAlgebra
using Statistics
using SpecialFunctions
using NLopt

export IMMCTM, MMCTM, ILDA, LDA, fit!, format_counts_lda, format_counts_ctm, format_counts_mmctm

include("IMMCTM.jl")
include("MMCTM.jl")
include("ILDA.jl")
include("LDA.jl")
include("common.jl")
include("utils.jl")

end # module
