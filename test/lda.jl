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
    Elnθ = [0.5  -1.1; 2.3 -0.7]
    Elnβ = [-0.2 -0.9; -1.1 0.3]

    ϕ = Array{Float64}(2, 2)
    # K1 W1
    ϕ[1, 1] = exp(Elnθ[1, 1] + Elnβ[1, 1])
    # K1 W2
    ϕ[1, 2] = exp(Elnθ[1, 1] + Elnβ[2, 1])
    # K2 W1
    ϕ[2, 1] = exp(Elnθ[2, 1] + Elnβ[1, 2])
    # K2 W2
    ϕ[2, 2] = exp(Elnθ[2, 1] + Elnβ[2, 2])

    ϕ[:, 1] ./= sum(ϕ[:, 1])
    ϕ[:, 2] ./= sum(ϕ[:, 2])

    model = MultiModalMuSig.LDA(K, α, η, X)
    model.Elnθ .= Elnθ
    model.Elnβ .= Elnβ
    MultiModalMuSig.update_ϕ!(model)

    @test model.ϕ[1] ≈ ϕ
end

@testset "update_γ!" begin
    ϕ = [0.4 0.2; 0.6 0.8]

    γ = Array{Float64}(2)
    γ[1] = α + (ϕ[1, 1] * X[1][1, 2]) + (ϕ[1, 2] * X[1][2, 2])
    γ[2] = α + (ϕ[2, 1] * X[1][1, 2]) + (ϕ[2, 2] * X[1][2, 2])

    Elnθ = Array{Float64}(2)
    Elnθ[1] = digamma(γ[1]) - digamma(γ[1] + γ[2])
    Elnθ[2] = digamma(γ[2]) - digamma(γ[1] + γ[2])

    model = MultiModalMuSig.LDA(K, α, η, X)
    model.ϕ[1] = ϕ
    MultiModalMuSig.update_γ!(model)

    @test model.γ[:, 1] ≈ γ
    @test model.Elnθ[:, 1] ≈ Elnθ
end

@testset "update_λ!" begin
    ϕ = [[0.4 0.2; 0.6 0.8], [0.1 0.6; 0.9 0.4]]

    λ = Array{Float64}(2, 2)
    λ[1, 1] = η + (ϕ[1][1, 1] * X[1][1, 2]) + (ϕ[2][1, 1] * X[2][1, 2])
    λ[2, 1] = η + (ϕ[1][1, 2] * X[1][2, 2]) + (ϕ[2][1, 2] * X[2][2, 2])
    λ[1, 2] = η + (ϕ[1][2, 1] * X[1][1, 2]) + (ϕ[2][2, 1] * X[2][1, 2])
    λ[2, 2] = η + (ϕ[1][2, 2] * X[1][2, 2]) + (ϕ[2][2, 2] * X[2][2, 2])

    Elnβ = Array{Float64}(2, 2)
    Elnβ[1, 1] = digamma(λ[1, 1]) - digamma(λ[1, 1] + λ[2, 1])
    Elnβ[2, 1] = digamma(λ[2, 1]) - digamma(λ[1, 1] + λ[2, 1])
    Elnβ[1, 2] = digamma(λ[1, 2]) - digamma(λ[1, 2] + λ[2, 2])
    Elnβ[2, 2] = digamma(λ[2, 2]) - digamma(λ[1, 2] + λ[2, 2])

    model = MultiModalMuSig.LDA(K, α, η, X)
    model.ϕ = ϕ
    MultiModalMuSig.update_λ!(model)

    @test model.λ ≈ λ
    @test model.Elnβ ≈ Elnβ
end
