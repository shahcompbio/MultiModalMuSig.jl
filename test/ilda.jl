using MultiModalMuSig
using Test

K = 2
α = 0.1
η = 0.1
features = [
    1 1;
    1 2;
    2 1;
    2 2
]
X = Matrix{Int}[
    [
        1 5;
        2 8
    ],
    [
        3 2;
        4 5
    ]
]

@testset "constructor" begin
    model = MultiModalMuSig.ILDA(K, α, η, features, X)
    @test model.K == 2
    @test model.D == 2
    @test model.I == 2
    @test model.J == [2, 2]

    @test model.η == [η, η]
    @test length(model.λ) == 2
    @test size(model.λ[1]) == (2, 2)
    @test size(model.λ[2]) == (2, 2)
    @test all(model.λ[1] .> 0) == true
    @test all(model.λ[2] .> 0) == true
    @test length(model.Elnβ) == 2
    @test size(model.Elnβ[1]) == (2, 2)
    @test size(model.Elnβ[2]) == (2, 2)

    @test model.α == α
    @test size(model.γ) == (2, 2)
    @test all(model.γ .> 0) == true
    @test size(model.Elnθ) == (2, 2)

    @test sum(model.ϕ[1], dims=1) ≈ ones(2)'

    model = MultiModalMuSig.ILDA(K, α, [0.01, 0.5], features, X)
    @test model.η == [0.01, 0.5]
end

@testset "update_ϕ!" begin
    Elnθ = [0.5  -1.1; 2.3 -0.7]
    Elnβ = [
        [-0.2 -0.9; -1.1 0.3],
        [0.5 0.1; -0.1 -0.4]
    ]

    model = MultiModalMuSig.ILDA(K, α, η, features, X)
    model.Elnθ .= Elnθ
    model.Elnβ .= Elnβ
    MultiModalMuSig.update_ϕ!(model)

    ϕ = Array{Float64}(undef, 2, 2)
    # K1 W1
    ϕ[1, 1] = exp(Elnθ[1, 1] + Elnβ[1][1, 1] + Elnβ[2][1, 1])
    # K1 W2
    ϕ[1, 2] = exp(Elnθ[1, 1] + Elnβ[1][1, 1] + Elnβ[2][2, 1])
    # K2 W1
    ϕ[2, 1] = exp(Elnθ[2, 1] + Elnβ[1][1, 2] + Elnβ[2][1, 2])
    # K2 W2
    ϕ[2, 2] = exp(Elnθ[2, 1] + Elnβ[1][1, 2] + Elnβ[2][2, 2])

    ϕ[:, 1] ./= sum(ϕ[:, 1])
    ϕ[:, 2] ./= sum(ϕ[:, 2])

    @test model.ϕ[1] ≈ ϕ

    # K1 W1
    ϕ[1, 1] = exp(Elnθ[1, 2] + Elnβ[1][2, 1] + Elnβ[2][1, 1])
    # K1 W2
    ϕ[1, 2] = exp(Elnθ[1, 2] + Elnβ[1][2, 1] + Elnβ[2][2, 1])
    # K2 W1
    ϕ[2, 1] = exp(Elnθ[2, 2] + Elnβ[1][2, 2] + Elnβ[2][1, 2])
    # K2 W2
    ϕ[2, 2] = exp(Elnθ[2, 2] + Elnβ[1][2, 2] + Elnβ[2][2, 2])

    ϕ[:, 1] ./= sum(ϕ[:, 1])
    ϕ[:, 2] ./= sum(ϕ[:, 2])

    @test model.ϕ[2] ≈ ϕ
end

@testset "update_γ!" begin
    ϕ = [0.4 0.2; 0.6 0.8]

    γ = Array{Float64}(undef, 2)
    γ[1] = α + (ϕ[1, 1] * X[1][1, 2]) + (ϕ[1, 2] * X[1][2, 2])
    γ[2] = α + (ϕ[2, 1] * X[1][1, 2]) + (ϕ[2, 2] * X[1][2, 2])

    Elnθ = Array{Float64}(undef, 2)
    Elnθ[1] = digamma(γ[1]) - digamma(γ[1] + γ[2])
    Elnθ[2] = digamma(γ[2]) - digamma(γ[1] + γ[2])

    model = MultiModalMuSig.ILDA(K, α, η, features, X)
    model.ϕ[1] = ϕ
    MultiModalMuSig.update_γ!(model)

    @test model.γ[:, 1] ≈ γ
    @test model.Elnθ[:, 1] ≈ Elnθ
end

@testset "update_λ!" begin
    η_test = [0.1, 0.2]
    ϕ = [[0.4 0.2; 0.6 0.8], [0.1 0.6; 0.9 0.4]]

    model = MultiModalMuSig.ILDA(K, α, η_test, features, X)
    model.ϕ = ϕ
    MultiModalMuSig.update_λ!(model)

    λ = Array{Float64}(undef, 2, 2)
    # I1
    # K1 J1
    λ[1, 1] = η_test[1] + (ϕ[1][1, 1] * X[1][1, 2]) + (ϕ[1][1, 2] * X[1][2, 2])
    # K1 J2
    λ[2, 1] = η_test[1] + (ϕ[2][1, 1] * X[2][1, 2]) + (ϕ[2][1, 2] * X[2][2, 2])
    # K2 J1
    λ[1, 2] = η_test[1] + (ϕ[1][2, 1] * X[1][1, 2]) + (ϕ[1][2, 2] * X[1][2, 2])
    # K2 J2
    λ[2, 2] = η_test[1] + (ϕ[2][2, 1] * X[2][1, 2]) + (ϕ[2][2, 2] * X[2][2, 2])

    Elnβ = Array{Float64}(undef, 2, 2)
    Elnβ[1, 1] = digamma(λ[1, 1]) - digamma(λ[1, 1] + λ[2, 1])
    Elnβ[2, 1] = digamma(λ[2, 1]) - digamma(λ[1, 1] + λ[2, 1])
    Elnβ[1, 2] = digamma(λ[1, 2]) - digamma(λ[1, 2] + λ[2, 2])
    Elnβ[2, 2] = digamma(λ[2, 2]) - digamma(λ[1, 2] + λ[2, 2])

    @test model.λ[1] ≈ λ
    @test model.Elnβ[1] ≈ Elnβ

    # I1
    # K1 J1
    λ[1, 1] = η_test[2] + (ϕ[1][1, 1] * X[1][1, 2]) + (ϕ[2][1, 1] * X[2][1, 2])
    # K1 J2
    λ[2, 1] = η_test[2] + (ϕ[1][1, 2] * X[1][2, 2]) + (ϕ[2][1, 2] * X[2][2, 2])
    # K2 J1
    λ[1, 2] = η_test[2] + (ϕ[1][2, 1] * X[1][1, 2]) + (ϕ[2][2, 1] * X[2][1, 2])
    # K2 J2
    λ[2, 2] = η_test[2] + (ϕ[1][2, 2] * X[1][2, 2]) + (ϕ[2][2, 2] * X[2][2, 2])

    Elnβ[1, 1] = digamma(λ[1, 1]) - digamma(λ[1, 1] + λ[2, 1])
    Elnβ[2, 1] = digamma(λ[2, 1]) - digamma(λ[1, 1] + λ[2, 1])
    Elnβ[1, 2] = digamma(λ[1, 2]) - digamma(λ[1, 2] + λ[2, 2])
    Elnβ[2, 2] = digamma(λ[2, 2]) - digamma(λ[1, 2] + λ[2, 2])

    @test model.λ[2] ≈ λ
    @test model.Elnβ[2] ≈ Elnβ
end

@testset "calculate_elbo" begin
    model = MultiModalMuSig.ILDA(K, α, η, features, X)

    @test typeof(MultiModalMuSig.calculate_ElnPβ(model)) == Float64
    @test typeof(MultiModalMuSig.calculate_ElnPθ(model)) == Float64
    @test typeof(MultiModalMuSig.calculate_ElnPZ(model)) == Float64
    @test typeof(MultiModalMuSig.calculate_ElnPX(model)) == Float64

    @test typeof(MultiModalMuSig.calculate_ElnQβ(model)) == Float64
    @test typeof(MultiModalMuSig.calculate_ElnQθ(model)) == Float64
    @test typeof(MultiModalMuSig.calculate_ElnQZ(model)) == Float64

    @test MultiModalMuSig.calculate_elbo(model) < 0.0
end
