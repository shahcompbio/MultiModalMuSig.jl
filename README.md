# MultiModalMuSig

[![Build Status](https://travis-ci.com/funnell/MultiModalMuSig.jl.svg?token=x1w6qwZCiTjuvfEXTgsz&branch=master)](https://travis-ci.com/funnell/MultiModalMuSig.jl) [![Coverage Status](https://coveralls.io/repos/funnell/MultiModalMuSig.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/funnell/MultiModalMuSig.jl?branch=master) [![codecov.io](http://codecov.io/github/funnell/MultiModalMuSig.jl/coverage.svg?branch=master)](http://codecov.io/github/funnell/MultiModalMuSig.jl?branch=master)

A Julia package implementing several topic models used for mutation signature estimation.

The Multi-modal correlated topic model (MMCTM) takes an array of arrays of matrices, where the first column of each matrix is a mutation type index, and the second column is the mutation count for a particular sample.

The following example shows how to perform inference using the MMCTM on SNV and SV counts:
```julia
using MultiModalMuSig
using CSV
using DataFrames
using VegaLite
using Random

Random.seed!(42)

snv_counts = CSV.read("data/brca-eu_snv_counts.tsv", delim='\t')
sv_counts = CSV.read("data/brca-eu_sv_counts.tsv", delim='\t')

X = format_counts_mmctm(snv_counts, sv_counts)
model = MMCTM([7, 7], [0.1, 0.1], X)
fit!(model, tol=1e-5)

snv_signatures = DataFrame(hcat(model.ϕ[1]...))
snv_signatures[:term] = snv_counts[:term]
snv_signatures = melt(
    snv_signatures, :term, variable_name=:signature, value_name=:probability
)
snv_signatures |> @vlplot(
    :bar, x={:term, sort=:null}, y=:probability, row=:signature,
    resolve={scale={y=:independent}}
)
```
[snv_signatures](https://user-images.githubusercontent.com/381464/47934375-8a8cec80-dead-11e8-8cfe-fbde1911ddc1.png)

Since these types of models can get stuck in poor local optima, it's a good idea to fit many models and pick the best one.
