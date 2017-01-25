using Base.Test

tests = ["mmctm"]

for test in tests
    include(test * ".jl")
end
