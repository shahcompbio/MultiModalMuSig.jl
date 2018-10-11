# MultiModalMuSig

[![Build Status](https://travis-ci.com/funnell/MultiModalMuSig.jl.svg?token=x1w6qwZCiTjuvfEXTgsz&branch=master)](https://travis-ci.com/funnell/MultiModalMuSig.jl)

[![Coverage Status](https://coveralls.io/repos/funnell/MultiModalMuSig.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/funnell/MultiModalMuSig.jl?branch=master)

[![codecov.io](http://codecov.io/github/funnell/MultiModalMuSig.jl/coverage.svg?branch=master)](http://codecov.io/github/funnell/MultiModalMuSig.jl?branch=master)

A Julia package implementing several topic models used for mutation signature estimation.

The Multi-modal correlated topic model (MMCTM) takes an array of arrays of matrices, where the first column of each matrix is a mutation type index, and the second column is the mutation count for a particular sample.
