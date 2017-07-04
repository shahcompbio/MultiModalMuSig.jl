using MultiModalMuSig
using Base.Test

K = [2, 3]
α = [0.1, 0.1]
features = Matrix{Int}[
    # modality 1
    [
        1 1;
        1 2;
        2 1;
        2 2
    ],
    # modality 2
    [
        1 1;
        1 2;
        2 1;
        2 2
    ]
]
X = Vector{Matrix{Int}}[
    # document 1
    [
        # modality 1
        [
            1 5;
            2 8
        ],
        # modality 2
        [
            1 2;
            2 5
        ]
    ],
    # document 2
    [
        # modality 1
        [
            3 4;
            4 9
        ],
        # modality 2
        [
            3 4;
            4 6
        ]
    ],
]

@testset "constructor" begin
    model = MultiModalMuSig.IMMCTM(K, α, features, X)
    @test model.D == 2
    @test model.N == [[13, 7], [13, 10]]
    @test model.M == 2
    @test model.I == [2, 2]
    @test model.J == [[2, 2], [2, 2]]
    @test model.V == [4, 4]

    @test length(model.μ) == 5
    @test size(model.Σ) == (5, 5)
    @test size(model.invΣ) == (5, 5)

    @test length(model.ζ) == 2
    @test length(model.ζ[1]) == 2
    @test sum(model.θ[1][1], 1) ≈ ones(2)'
    @test length(model.λ[1]) == 5
    @test model.ν[1] == ones(5)

    @test length(model.γ) == 2
    @test length(model.γ[1]) == 2
    @test length(model.γ[1][1]) == 2
    @test length(model.γ[1][1][1]) == 2
    @test all(model.γ[1][1][1] .> 0) == true
end

@testset "calculate_Ndivζ" begin
    model = MultiModalMuSig.IMMCTM(K, α, features, X)
    model.ζ = [[2, 3], [4,5]]

    Ndivζ = [
        sum(X[1][1][:, 2]) / model.ζ[1][1],
        sum(X[1][1][:, 2]) / model.ζ[1][1],
        sum(X[1][2][:, 2]) / model.ζ[1][2],
        sum(X[1][2][:, 2]) / model.ζ[1][2],
        sum(X[1][2][:, 2]) / model.ζ[1][2]
    ]

    @test MultiModalMuSig.calculate_Ndivζ(model, 1) ≈ Ndivζ
end

@testset "calculate_sumθ" begin
    model = MultiModalMuSig.IMMCTM(K, α, features, X)
    θ = Matrix{Float64}[]
    push!(θ, [0.4 0.1; 0.6 0.9])
    push!(θ, [0.3 0.4; 0.3 0.5; 0.4 0.1])
    model.θ[1] = θ

    sumθ = [
        X[1][1][1, 2] * θ[1][1, 1] + X[1][1][2, 2] * θ[1][1, 2],
        X[1][1][1, 2] * θ[1][2, 1] + X[1][1][2, 2] * θ[1][2, 2],
        X[1][2][1, 2] * θ[2][1, 1] + X[1][2][2, 2] * θ[2][1, 2],
        X[1][2][1, 2] * θ[2][2, 1] + X[1][2][2, 2] * θ[2][2, 2],
        X[1][2][1, 2] * θ[2][3, 1] + X[1][2][2, 2] * θ[2][3, 2],
    ]

    @test MultiModalMuSig.calculate_sumθ(model, 1) ≈ sumθ
end

@testset "update_λ!" begin
    model = MultiModalMuSig.IMMCTM(K, α, features, X)

    λ = Float64[1, 2, 3, 4, 1]
    model.λ[1] = copy(λ)

    MultiModalMuSig.update_λ!(model, 1)
    @test model.λ[1] ≉ λ
end

@testset "ν_objective" begin
    model = MultiModalMuSig.IMMCTM(K, α, features, X)

    μ = Float64[1, 1, 2, 2, 1]
    invΣ = eye(sum(K))
    λ = Float64[1, 2, 3, 4, 1]
    ν = Float64[1, 1, 1, 2, 1]
    ζ = Float64[2, 1]

    L = (
        -0.5 * trace(diagm(ν) * invΣ) -
        sum(X[1][1][:, 2]) / ζ[1] *
            (exp(λ[1] + 0.5ν[1]) + exp(λ[2] + 0.5ν[2])) -
        sum(X[1][2][:, 2]) / ζ[2] *
            (exp(λ[3] + 0.5ν[3]) + exp(λ[4] + 0.5ν[4]) + exp(λ[5] + 0.5ν[5])) +
        0.5 * (log(ν[1]) + log(ν[2]) + log(ν[3]) + log(ν[4]) + log(ν[5]))
    )
    ∇ν = Array{Float64}(sum(K))
    Ndivζ = vcat([fill(model.N[1][m] / ζ[m], K[m]) for m in 1:model.M]...)
    objective = MultiModalMuSig.ν_objective(ν, ∇ν, λ, Ndivζ, μ, invΣ)
    @test objective ≈ L

    grad = -0.5 * diag(invΣ)
    grad[1] += (
        -sum(X[1][1][:, 2]) / (2 * ζ[1]) * exp(λ[1] + 0.5ν[1]) + 1 / (2ν[1])
    )
    grad[2] += (
        -sum(X[1][1][:, 2]) / (2 * ζ[1]) * exp(λ[2] + 0.5ν[2]) + 1 / (2ν[2])
    )
    grad[3] += (
        -sum(X[1][2][:, 2]) / (2 * ζ[2]) * exp(λ[3] + 0.5ν[3]) + 1 / (2ν[3])
    )
    grad[4] += (
        -sum(X[1][2][:, 2]) / (2 * ζ[2]) * exp(λ[4] + 0.5ν[4]) + 1 / (2ν[4])
    )
    grad[5] += (
        -sum(X[1][2][:, 2]) / (2 * ζ[2]) * exp(λ[5] + 0.5ν[5]) + 1 / (2ν[5])
    )
    @test ∇ν ≈ grad

    model.μ = μ
    model.λ[1] = λ
    model.ν[1] = ν
    model.ζ[1] = ζ
    MultiModalMuSig.update_ν!(model, 1)
    @test (model.ν[1] .> 0.0) == fill(true, sum(K))
    @test (model.λ[1] .< 100.0) == fill(true, sum(K))
end

@testset "update_ζ!" begin
    model = MultiModalMuSig.IMMCTM(K, α, features, X)
    model.λ = [[1,2,3,4,1], [2,3,1,4,2]]
    model.ν = [[1,1,1,2,1], [1,3,1,2,1]]
    MultiModalMuSig.update_ζ!(model, 1)

    ζ = Float64[exp(1.5) + exp(2.5), exp(3.5) + exp(5) + exp(1.5)]
    @test model.ζ[1] ≈ ζ
end

@testset "update_θ!" begin
    model = MultiModalMuSig.IMMCTM(K, α, features, X)
    model.λ = [[1,2,3,4,1], [2,3,1,4,2]]
    model.γ = [
        [
            [[0.1, 0.2],[0.1, 1.0]],
            [[0.1, 0.1],[1.0, 1.0]]
        ],
        [
            [[0.5, 0.5],[1.0, 1.5]],
            [[1.0, 2.0],[2.0, 3.0]],
            [[1.0, 5.0],[5.0, 2.0]]
        ]
    ]
    MultiModalMuSig.update_Elnϕ!(model)
    MultiModalMuSig.update_θ!(model, 1)

    θ = Array{Float64}(2, 2)
    θ[1, 1] = exp(1 + digamma(0.1) - digamma(0.3) + digamma(0.1) - digamma(1.1))
    θ[2, 1] = exp(2 + digamma(0.1) - digamma(0.2) + digamma(1.0) - digamma(2.0))
    θ[1, 2] = exp(1 + digamma(0.1) - digamma(0.3) + digamma(1.0) - digamma(1.1))
    θ[2, 2] = exp(2 + digamma(0.1) - digamma(0.2) + digamma(1.0) - digamma(2.0))
    θ[:, 1] ./= sum(θ[:, 1])
    θ[:, 2] ./= sum(θ[:, 2])

    @test sum(model.θ[1][1], 1) ≈ ones(size(X[1][1])[1])'
    @test model.θ[1][1] ≈ θ
    @test any(model.θ[1][1] .< 0.0) == false

    MultiModalMuSig.update_θ!(model, 2)
    θ = Array{Float64}(3, 2)
    θ[1, 1] = exp(1 + digamma(0.5) - digamma(1.0) + digamma(1.0) - digamma(2.5))
    θ[2, 1] = exp(4 + digamma(2.0) - digamma(3.0) + digamma(2.0) - digamma(5.0))
    θ[3, 1] = exp(2 + digamma(5.0) - digamma(6.0) + digamma(5.0) - digamma(7.0))
    θ[1, 2] = exp(1 + digamma(0.5) - digamma(1.0) + digamma(1.5) - digamma(2.5))
    θ[2, 2] = exp(4 + digamma(2.0) - digamma(3.0) + digamma(3.0) - digamma(5.0))
    θ[3, 2] = exp(2 + digamma(5.0) - digamma(6.0) + digamma(2.0) - digamma(7.0))
    θ[:, 1] ./= sum(θ[:, 1])
    θ[:, 2] ./= sum(θ[:, 2])

    @test model.θ[2][2] ≈ θ
end

@testset "update_μ!" begin
    model = MultiModalMuSig.IMMCTM(K, α, features, X)
    model.λ = [[1,2,3,4,1], [2,3,1,4,2]]

    MultiModalMuSig.update_μ!(model)

    @test model.μ ≈ [1.5, 2.5, 2.0, 4.0, 1.5]
end

@testset "update_Σ!" begin
    model = MultiModalMuSig.IMMCTM(K, α, features, X)
    model.λ = [[1,2,3,4,1], [2,3,1,4,2]]
    model.ν = [[1,1,1,2,1], [1,3,1,2,1]]
    model.μ = [1, 1, 2, 2, 1]

    MultiModalMuSig.update_Σ!(model)

    diff1 = model.λ[1] .- model.μ
    diff2 = model.λ[2] .- model.μ
    Σ = 0.5 * (
        diagm(model.ν[1]) + diagm(model.ν[2]) +
        (diff1 * diff1') + (diff2 * diff2')
    )
    @test model.Σ ≈ Σ
    @test model.invΣ ≈ inv(Σ)
end

@testset "update_γ!" begin
    model = MultiModalMuSig.IMMCTM(K, α, features, X)
    model.θ[1][1] = [0.4 0.1; 0.6 0.9]
    model.θ[2][1] = [0.3 0.5; 0.7 0.5]
    MultiModalMuSig.update_γ!(model)

    γ1 = [0.1 + 5 * 0.4 + 8 * 0.1, 0.1 + 4 * 0.3 + 9 * 0.5]
    γ2 = [0.1 + 5 * 0.4 + 4 * 0.3, 0.1 + 8 * 0.1 + 9 * 0.5]
    @test model.γ[1][1][1] ≈ γ1
    @test model.γ[1][1][2] ≈ γ2
end

@testset "update_Elnϕ!" begin
    model = MultiModalMuSig.IMMCTM(K, α, features, X)
    model.γ[1][1][1] .= [1, 2]

    MultiModalMuSig.update_Elnϕ!(model)

    @test model.Elnϕ[1][1][1][1] ≈ digamma(1) - digamma(3)
end

# TODO
@testset "update_α!" begin
    model = MultiModalMuSig.IMMCTM(K, α, features, X)

    sum_Elnϕ = sum(model.Elnϕ[1][1][1]) + sum(model.Elnϕ[1][2][1])
    L = K[1] * (lgamma(2α[1]) - 2lgamma(α[1])) + α[1] * sum_Elnϕ
    grad = 2K[1] * (digamma(2α[1]) - digamma(α[1])) + sum_Elnϕ

    ∇α = Array{Float64}(1)
    @test MultiModalMuSig.α_objective(model.α[1][1:1], ∇α, sum_Elnϕ, K[1], 2) ≈ L
    @test ∇α[1] ≈ grad

    sum_Elnϕ = sum(model.Elnϕ[2][1][2]) + sum(model.Elnϕ[2][2][2])
    L_before = K[2] * (lgamma(2α[2]) - 2lgamma(α[2])) + α[2] * sum_Elnϕ
    MultiModalMuSig.update_α!(model)
    L_after = (
        K[2] * (lgamma(2model.α[2][2]) - 2lgamma(model.α[2][2])) +
        model.α[2][2] * sum_Elnϕ
    )
    @test model.α[1] ≉ fill(α[1], 2)
    @test model.α[2] ≉ fill(α[2], 2)
    @test L_after > L_before
end

@testset "calculate_ElnPϕ" begin
    model = MultiModalMuSig.IMMCTM(K, α, features, X)
    ElnPϕ = 0.0
    #@test MultiModalMuSig.calculate_ElnPϕ(model) ≈ ElnPϕ
end

@testset "calculate_ElnPη" begin
    model = MultiModalMuSig.IMMCTM(K, α, features, X)
    ElnPη = 0.0
    #@test MultiModalMuSig.calculate_ElnPη(model) ≈ ElnPη
end

@testset "calculate_ElnPZ" begin
    model = MultiModalMuSig.IMMCTM(K, α, features, X)
    ElnPZ = 0.0
    #@test MultiModalMuSig.calculate_ElnPZ(model) ≈ ElnPZ
end

@testset "calculate_ElnPX" begin
    model = MultiModalMuSig.IMMCTM(K, α, features, X)
    ElnPX = 0.0
    #@test MultiModalMuSig.calculate_ElnPX(model) ≈ ElnPX
end

@testset "calculate_ElnQϕ" begin
    model = MultiModalMuSig.IMMCTM(K, α, features, X)
    ElnQϕ = 0.0
    #@test MultiModalMuSig.calculate_ElnQϕ(model) ≈ ElnQϕ
end

@testset "calculate_ElnQη" begin
    model = MultiModalMuSig.IMMCTM(K, α, features, X)
    ElnQη = 0.0
    #@test MultiModalMuSig.calculate_ElnQη(model) ≈ ElnQη
end

@testset "calculate_ElnQZ" begin
    model = MultiModalMuSig.IMMCTM(K, α, features, X)
    ElnQZ = 0.0
    #@test MultiModalMuSig.calculate_ElnQZ(model) ≈ ElnQZ
end

@testset "calculate_elbo" begin
    model = MultiModalMuSig.IMMCTM(K, α, features, X)
    @test MultiModalMuSig.calculate_elbo(model) < 0.0
end

@testset "fit" begin
    model = MultiModalMuSig.IMMCTM(K, α, features, X)
    ll = MultiModalMuSig.fit!(model, maxiter=1, verbose=false) 
    @test length(ll) == 1
    @test length(ll[1]) == 2
end

@testset "loglikelihoods"  begin
    η = [[1.0, 2.0], [2.0, 3.0]]
    θ = [exp.(η[d]) ./ sum(exp.(η[d])) for d in 1:2]

    γ = [
        [[0.1, 0.2],[0.1, 1.0]],
        [[0.1, 0.1],[1.0, 1.0]]
    ]
    ϕ = [[γ[k][i] ./ sum(γ[k][i]) for i in 1:2] for k in 1:2]

    Xm1 = [X[d][1] for d in 1:2]

    sum_ll = (
        Xm1[1][1, 2] * log(
            θ[1][1] * ϕ[1][1][1] * ϕ[1][2][1] +
            θ[1][2] * ϕ[2][1][1] * ϕ[2][2][1]
        ) +
        Xm1[1][2, 2] * log(
            θ[1][1] * ϕ[1][1][1] * ϕ[1][2][2] +
            θ[1][2] * ϕ[2][1][1] * ϕ[2][2][2]
        ) +
        Xm1[2][1, 2] * log(
            θ[2][1] * ϕ[1][1][2] * ϕ[1][2][1] +
            θ[2][2] * ϕ[2][1][2] * ϕ[2][2][1]
        ) +
        Xm1[2][2, 2] * log(
            θ[2][1] * ϕ[1][1][2] * ϕ[1][2][2] +
            θ[2][2] * ϕ[2][1][2] * ϕ[2][2][2]
        )
    )
    ll = sum_ll / sum(sum(X[d][1][:, 2]) for d in 1:2)
    res = MultiModalMuSig.calculate_modality_loglikelihood(
        Xm1, η, ϕ, features[1]
    )

    @test res ≈ ll
end
