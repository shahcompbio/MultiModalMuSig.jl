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

    ∇λ = Array{Float64}(sum(K))
    sumθ = vcat([vec(sum(θ[m] .* X[1][m][:, 2]', 2)) for m in 1:model.M]...)
    Ndivζ = vcat([fill(model.N[1][m] / ζ[m], K[m]) for m in 1:model.M]...)
    objective = MultiModalMuSig.λ_objective(λ, ∇λ, ν, Ndivζ, sumθ, μ, invΣ)
    @test objective ≈ calc_λ_obj(μ, invΣ, λ, ν, ζ, θ, X)
    @test ∇λ ≈ calc_λ_grad(μ, invΣ, λ, ν, ζ, θ, X)
end
