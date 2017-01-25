using MultiModalMuSig
using Base.Test

K = [2, 3]
α = [0.1, 0.1]
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
    model = MultiModalMuSig.MMCTM(K, α, X)
    @test model.D == 2
    @test model.N == [[13, 7], [13, 10]]
    @test model.M == 2
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
    @test length(model.γ[1][1]) == 4
    @test length(model.γ[1][2]) == 4
end

@testset "calculate_Ndivζ" begin
    model = MultiModalMuSig.MMCTM(K, α, X)
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
    model = MultiModalMuSig.MMCTM(K, α, X)
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

function calc_λ_obj(μ, invΣ, λ, ν, ζ, θ, X)
    diff = λ .- μ
    L = (
        -0.5 * (diff' * invΣ * diff)[1] +
        X[1][1][1, 2] * (θ[1][1, 1] * λ[1] + θ[1][2, 1] * λ[2]) +
        X[1][1][2, 2] * (θ[1][1, 2] * λ[1] + θ[1][2, 2] * λ[2]) +
        X[1][2][1, 2] *
            (θ[2][1, 1] * λ[3] + θ[2][2, 1] * λ[4] + θ[2][3, 1] * λ[5]) +
        X[1][2][2, 2] *
            (θ[2][1, 2] * λ[3] + θ[2][2, 2] * λ[4] + θ[2][3, 2] * λ[5]) -
        sum(X[1][1][:, 2]) / ζ[1] *
            (exp(λ[1] + 0.5ν[1]) + exp(λ[2] + 0.5ν[2])) -
        sum(X[1][2][:, 2]) / ζ[2] *
            (exp(λ[3] + 0.5ν[3]) + exp(λ[4] + 0.5ν[4]) + exp(λ[5] + 0.5ν[5]))
    )
    return L
end

function calc_λ_grad(μ, invΣ, λ, ν, ζ, θ, X)
    diff = λ .- μ
    grad = -invΣ * diff
    grad[1] += (
        X[1][1][1, 2] * θ[1][1, 1] + X[1][1][2, 2] * θ[1][1, 2]
        - sum(X[1][1][:, 2]) / ζ[1] * exp(λ[1] + 0.5ν[1])
    )
    grad[2] += (
        X[1][1][1, 2] * θ[1][2, 1] + X[1][1][2, 2] * θ[1][2, 2]
        - sum(X[1][1][:, 2]) / ζ[1] * exp(λ[2] + 0.5ν[2])
    )
    grad[3] += (
        X[1][2][1, 2] * θ[2][1, 1] + X[1][2][2, 2] * θ[2][1, 2]
        - sum(X[1][2][:, 2]) / ζ[2] * exp(λ[3] + 0.5ν[3])
    )
    grad[4] += (
        X[1][2][1, 2] * θ[2][2, 1] + X[1][2][2, 2] * θ[2][2, 2]
        - sum(X[1][2][:, 2]) / ζ[2] * exp(λ[4] + 0.5ν[4])
    )
    grad[5] += (
        X[1][2][1, 2] * θ[2][3, 1] + X[1][2][2, 2] * θ[2][3, 2]
        - sum(X[1][2][:, 2]) / ζ[2] * exp(λ[5] + 0.5ν[5])
    )
    return grad
end

@testset "λ_objective" begin
    model = MultiModalMuSig.MMCTM(K, α, X)

    μ = Float64[1, 1, 2, 2, 1]
    invΣ = model.invΣ
    λ = Float64[1, 2, 3, 4, 1]
    ν = Float64[1, 1, 1, 2, 1]
    ζ = Float64[2, 1]
    θ = Matrix{Float64}[]
    push!(θ, [0.4 0.1; 0.6 0.9])
    push!(θ, [0.3 0.4; 0.3 0.5; 0.4 0.1])

    ∇λ = Array(Float64, sum(K))
    sumθ = vcat([vec(sum(θ[m] .* X[1][m][:, 2]', 2)) for m in 1:model.M]...)
    Ndivζ = vcat([fill(model.N[1][m] / ζ[m], K[m]) for m in 1:model.M]...)
    objective = MultiModalMuSig.λ_objective(λ, ∇λ, ν, Ndivζ, sumθ, μ, invΣ)
    @test objective ≈ calc_λ_obj(μ, invΣ, λ, ν, ζ, θ, X)
    @test ∇λ ≈ calc_λ_grad(μ, invΣ, λ, ν, ζ, θ, X)

    model.μ .= μ
    model.λ[1] .= λ
    model.ν[1] .= ν
    model.ζ[1] .= ζ
    model.θ[1] = deepcopy(θ)
    MultiModalMuSig.update_λ!(model, 1)
    @test isnan(model.λ[1]) == fill(false, sum(K))
end

function calc_ν_L(invΣ, λ, ν, ζ, X)
    L = (
        -0.5 * trace(diagm(ν) * invΣ) -
        sum(X[1][1][:, 2]) / ζ[1] *
            (exp(λ[1] + 0.5ν[1]) + exp(λ[2] + 0.5ν[2])) -
        sum(X[1][2][:, 2]) / ζ[2] *
            (exp(λ[3] + 0.5ν[3]) + exp(λ[4] + 0.5ν[4]) + exp(λ[5] + 0.5ν[5])) +
        0.5 * (log(ν[1]) + log(ν[2]) + log(ν[3]) + log(ν[4]) + log(ν[5]))
    )
    return L
end

function calc_ν_grad(invΣ, λ, ν, ζ, X)
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
    return grad
end

@testset "ν_objective" begin
    model = MultiModalMuSig.MMCTM(K, α, X)

    μ = Float64[1, 1, 2, 2, 1]
    invΣ = eye(sum(K))
    λ = Float64[1, 2, 3, 4, 1]
    ν = Float64[1, 1, 1, 2, 1]
    ζ = Float64[2, 1]

    ∇ν = Array(Float64, sum(K))
    Ndivζ = vcat([fill(model.N[1][m] / ζ[m], K[m]) for m in 1:model.M]...)
    objective = MultiModalMuSig.ν_objective(ν, ∇ν, λ, Ndivζ, μ, invΣ)
    @test objective ≈ calc_ν_L(invΣ, λ, ν, ζ, X)
    @test ∇ν ≈ calc_ν_grad(invΣ, λ, ν, ζ, X)

    model.μ = μ
    model.λ[1] = λ
    model.ν[1] = ν
    model.ζ[1] = ζ
    MultiModalMuSig.update_ν!(model, 1)
    @test (model.ν[1] .> 0.0) == fill(true, sum(K))
end

@testset "update_ζ!" begin
    model = MultiModalMuSig.MMCTM(K, α, X)
    model.λ = [[1,2,3,4,1], [2,3,1,4,2]]
    model.ν = [[1,1,1,2,1], [1,3,1,2,1]]
    MultiModalMuSig.update_ζ!(model, 1)

    ζ = Float64[exp(1.5) + exp(2.5), exp(3.5) + exp(5) + exp(1.5)]
    @test model.ζ[1] ≈ ζ
end

@testset "update_θ!" begin
    model = MultiModalMuSig.MMCTM(K, α, X)
    model.λ = [[1,2,3,4,1], [2,3,1,4,2]]
    model.γ = [
        [
            [1, 2, 2, 6],
            [2, 3, 1, 2]
        ],
        [
            [1, 2, 3, 4],
            [2, 1, 2, 6],
            [1, 1, 3, 1]
        ]
    ]
    MultiModalMuSig.update_Elnϕ!(model)
    MultiModalMuSig.update_θ!(model, 1)

    θ = Array(Float64, 2, 2)
    θ[1, 1] = exp(1 + digamma(1) - digamma(11))
    θ[2, 1] = exp(2 + digamma(2) - digamma(8))
    θ[1, 2] = exp(1 + digamma(2) - digamma(11))
    θ[2, 2] = exp(2 + digamma(3) - digamma(8))
    θ[:, 1] ./= sum(θ[:, 1])
    θ[:, 2] ./= sum(θ[:, 2])

    @test sum(model.θ[1][1], 1) ≈ ones(size(X[1][1])[1])'
    @test model.θ[1][1] ≈ θ
    @test any(model.θ[1][1] .< 0.0) == false

    MultiModalMuSig.update_θ!(model, 2)
    θ = Array(Float64, 3, 2)
    θ[1, 1] = exp(1 + digamma(3) - digamma(10))
    θ[2, 1] = exp(4 + digamma(2) - digamma(11))
    θ[3, 1] = exp(2 + digamma(3) - digamma(6))
    θ[1, 2] = exp(1 + digamma(4) - digamma(10))
    θ[2, 2] = exp(4 + digamma(6) - digamma(11))
    θ[3, 2] = exp(2 + digamma(1) - digamma(6))
    θ[:, 1] ./= sum(θ[:, 1])
    θ[:, 2] ./= sum(θ[:, 2])

    @test model.θ[2][2] ≈ θ
end

@testset "update_μ!" begin
    model = MultiModalMuSig.MMCTM(K, α, X)
    model.λ = [[1,2,3,4,1], [2,3,1,4,2]]

    MultiModalMuSig.update_μ!(model)

    @test model.μ ≈ [1.5, 2.5, 2.0, 4.0, 1.5]
end

@testset "update_Σ!" begin
    model = MultiModalMuSig.MMCTM(K, α, X)
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
    model = MultiModalMuSig.MMCTM(K, α, X)
    model.θ[1][1] = [0.4 0.1; 0.6 0.9]
    model.θ[2][1] = [0.3 0.5; 0.7 0.5]
    model.θ[1][2] = [0.2 0.6; 0.7 0.3; 0.1 0.1]
    model.θ[2][2] = [0.1 0.3; 0.7 0.5; 0.2 0.2]
    MultiModalMuSig.update_γ!(model)

    γ1 = [0.1 + 5 * 0.4, 0.1 + 8 * 0.1, 0.1 + 4 * 0.3, 0.1 + 9 * 0.5]
    γ2 = [0.1 + 5 * 0.6, 0.1 + 8 * 0.9, 0.1 + 4 * 0.7, 0.1 + 9 * 0.5]
    @test model.γ[1][1] ≈ γ1
    @test model.γ[1][2] ≈ γ2

    γ1 = [0.1 + 2 * 0.2, 0.1 + 5 * 0.6, 0.1 + 4 * 0.1, 0.1 + 6 * 0.3]
    γ2 = [0.1 + 2 * 0.7, 0.1 + 5 * 0.3, 0.1 + 4 * 0.7, 0.1 + 6 * 0.5]
    γ3 = [0.1 + 2 * 0.1, 0.1 + 5 * 0.1, 0.1 + 4 * 0.2, 0.1 + 6 * 0.2]
    @test model.γ[2][1] ≈ γ1
    @test model.γ[2][2] ≈ γ2
    @test model.γ[2][3] ≈ γ3
end

@testset "update_Elnϕ!" begin
    model = MultiModalMuSig.MMCTM(K, α, X)
    model.γ[1][1] .= [1, 2, 1, 3]

    MultiModalMuSig.update_Elnϕ!(model)

    @test model.Elnϕ[1][1][1] ≈ digamma(1) - digamma(7)
end

#@testset "calculate_ElnPϕ" begin
    #model = MultiModalMuSig.MMCTM(K, α, X)
    #ElnPϕ = 0.0
    #@pending @test MultiModalMuSig.calculate_ElnPϕ(model) ≈ ElnPϕ
#end

#@testset "calculate_ElnPη" begin
    #model = MultiModalMuSig.MMCTM(K, α, X)
    #ElnPη = 0.0
    #@pending @test MultiModalMuSig.calculate_ElnPη(model) ≈ ElnPη
#end

#@testset "calculate_ElnPZ" begin
    #model = MultiModalMuSig.MMCTM(K, α, X)
    #ElnPZ = 0.0
    #@pending @test MultiModalMuSig.calculate_ElnPZ(model) ≈ ElnPZ
#end

#@testset "calculate_ElnPX" begin
    #model = MultiModalMuSig.MMCTM(K, α, X)
    #ElnPX = 0.0
    #@pending @test MultiModalMuSig.calculate_ElnPX(model) ≈ ElnPX
#end

#@testset "calculate_ElnQϕ" begin
    #model = MultiModalMuSig.MMCTM(K, α, X)
    #ElnQϕ = 0.0
    #@pending @test MultiModalMuSig.calculate_ElnQϕ(model) ≈ ElnQϕ
#end

#@testset "calculate_ElnQη" begin
    #model = MultiModalMuSig.MMCTM(K, α, X)
    #ElnQη = 0.0
    #@pending @test MultiModalMuSig.calculate_ElnQη(model) ≈ ElnQη
#end

#@testset "calculate_ElnQZ" begin
    #model = MultiModalMuSig.MMCTM(K, α, X)
    #ElnQZ = 0.0
    #@pending @test MultiModalMuSig.calculate_ElnQZ(model) ≈ ElnQZ
#end

@testset "calculate_elbo" begin
    model = MultiModalMuSig.MMCTM(K, α, X)
    @test MultiModalMuSig.calculate_elbo(model) <= 0.0
end

@testset "fit" begin
    model = MultiModalMuSig.MMCTM(K, α, X)
    ll = MultiModalMuSig.fit!(model, maxiter=1, verbose=false) 
    @test length(ll) == 1
    @test length(ll[1]) == 2
end

@testset "loglikelihoods" begin 
    η = [[1.0, 2.0], [2.0, 3.0]]
    θ = [exp(η[d]) ./ sum(exp(η[d])) for d in 1:2]

    γ = [
        [1, 2, 1, 3],
        [1, 1, 2, 4]
    ]
    ϕ = [γ[k] ./ sum(γ[k]) for k in 1:2]

    Xm1 = [X[d][1] for d in 1:2]

    sum_ll = [
        (
            Xm1[1][1, 2] * log(θ[1][1] * ϕ[1][1] + θ[1][2] * ϕ[2][1]) +
            Xm1[1][2, 2] * log(θ[1][1] * ϕ[1][2] + θ[1][2] * ϕ[2][2])
        ),
        (
            Xm1[2][1, 2] * log(θ[2][1] * ϕ[1][3] + θ[2][2] * ϕ[2][3]) +
            Xm1[2][2, 2] * log(θ[2][1] * ϕ[1][4] + θ[2][2] * ϕ[2][4])
        )
    ]
    N = [sum(X[d][1][:, 2]) for d in 1:2]

    res = MultiModalMuSig.calculate_docmodality_loglikelihood(Xm1[1], η[1], ϕ)
    @test res ≈ sum_ll[1] / N[1]

    res = MultiModalMuSig.calculate_modality_loglikelihood(Xm1, η, ϕ)
    @test res ≈ sum(sum_ll) / sum(N)

    model = MultiModalMuSig.MMCTM(K, α, X)
    model.λ[1][1:2] .= η[1][1:2]
    model.λ[2][1:2] .= η[2][1:2]
    model.γ[1] = deepcopy(γ)

    res = MultiModalMuSig.calculate_loglikelihoods(X, model)
    @test res[1] ≈ sum(sum_ll) / sum(N)
end
