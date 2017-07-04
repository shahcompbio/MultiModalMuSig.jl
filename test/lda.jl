using MultiModalMuSig
using Base.Test

K = 2
α = 0.1
η = 0.1
X = Matrix{Int}[
    [
        1 5;
        2 8
    ],
    [
        1 2;
        2 5
    ]
]

@testset "constructor" begin
    model = MultiModalMuSig.LDA(K, α, η, X)
    @test model.K == 2
    @test model.D == 2
    @test model.N == [13, 7]
    @test model.V == 2

    @test size(model.λ) == (2, 2)
    @test all(model.λ .> 0) == true

    @test size(model.γ) == (2, 2)
    @test all(model.γ .> 0) == true

    @test sum(model.ϕ[1], 1) ≈ ones(2)'

    model = MultiModalMuSig.LDA(K, α, η, 3, X)
    @test model.V == 3
    @test size(model.λ) == (3, 2)
end

@testset "update_ϕ!" begin
    Elnγ = [0.5  -1.1; 2.3 -0.7]
    Elnλ = [-0.2 -0.9; -1.1 0.3]

    ϕ = Array{Float64}(2, 2)
    # K1 W1
    ϕ[1, 1] = exp(Elnγ[1, 1] + Elnλ[1, 1])
    # K1 W2
    ϕ[1, 2] = exp(Elnγ[1, 1] + Elnλ[2, 1])
    # K2 W1
    ϕ[2, 1] = exp(Elnγ[2, 1] + Elnλ[1, 2])
    # K2 W2
    ϕ[2, 2] = exp(Elnγ[2, 1] + Elnλ[2, 2])

    ϕ[:, 1] ./= sum(ϕ[:, 1])
    ϕ[:, 2] ./= sum(ϕ[:, 2])

    model = MultiModalMuSig.LDA(K, α, η, X)
    model.Elnγ .= Elnγ
    model.Elnλ .= Elnλ
    MultiModalMuSig.update_ϕ!(model)

    @test model.ϕ[1] ≈ ϕ
end
