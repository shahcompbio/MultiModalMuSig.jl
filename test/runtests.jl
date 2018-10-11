using Test

tests = ["immctm", "mmctm", "ilda", "lda", "common"]

for test in tests
    include(test * ".jl")
end
