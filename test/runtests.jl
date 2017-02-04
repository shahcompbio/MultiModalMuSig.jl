using Base.Test

tests = ["immctm", "mmctm"]

for test in tests
    include(test * ".jl")
end
