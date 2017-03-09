using Base.Test

tests = ["immctm", "mmctm", "common"]

for test in tests
    include(test * ".jl")
end
