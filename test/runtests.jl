using Base.Test

tests = ["immctm", "mmctm", "lda", "common"]

for test in tests
    include(test * ".jl")
end
