module MultiModalMuSigTests

using MultiModalMuSig
using FactCheck

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

facts("constructor") do
    model = MultiModalMuSig.IMMCTM(K, α, features, X)
    @fact model.D --> 2
    @fact model.N --> [[13, 7], [13, 10]]
    @fact model.M --> 2
    @fact model.I --> [2, 2]
    @fact model.J --> [[2, 2], [2, 2]]
    @fact model.V --> [4, 4]

    @fact length(model.μ) --> 5
    @fact size(model.Σ) --> (5, 5)
    @fact size(model.invΣ) --> (5, 5)

    @fact length(model.ζ) --> 2
    @fact length(model.ζ[1]) --> 2
    @fact sum(model.θ[1][1], 1) --> roughly(ones(2)')
    @fact length(model.λ[1]) --> 5
    @fact model.ν[1] --> ones(5)

    @fact length(model.γ) --> 2
    @fact length(model.γ[1]) --> 2
    @fact length(model.γ[1][1]) --> 2
    @fact length(model.γ[1][1][1]) --> 2
end

facts("calculate_Ndivζ") do
    model = MultiModalMuSig.IMMCTM(K, α, features, X)
    model.ζ = [[2, 3], [4,5]]

    Ndivζ = [
        sum(X[1][1][:, 2]) / model.ζ[1][1],
        sum(X[1][1][:, 2]) / model.ζ[1][1],
        sum(X[1][2][:, 2]) / model.ζ[1][2],
        sum(X[1][2][:, 2]) / model.ζ[1][2],
        sum(X[1][2][:, 2]) / model.ζ[1][2]
    ]

    @fact MultiModalMuSig.calculate_Ndivζ(model, 1) --> roughly(Ndivζ)
end

facts("calculate_sumθ") do
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

    @fact MultiModalMuSig.calculate_sumθ(model, 1) --> roughly(sumθ)
end

facts("λ_objective") do
    model = MultiModalMuSig.IMMCTM(K, α, features, X)

    μ = Float64[1, 1, 2, 2, 1]
    invΣ = model.invΣ
    λ = Float64[1, 2, 3, 4, 1]
    ν = Float64[1, 1, 1, 2, 1]
    ζ = Float64[2, 1]
    θ = Matrix{Float64}[]
    push!(θ, [0.4 0.1; 0.6 0.9])
    push!(θ, [0.3 0.4; 0.3 0.5; 0.4 0.1])

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
    ∇λ = Array(Float64, sum(K))
    sumθ = vcat([vec(sum(θ[m] .* X[1][m][:, 2]', 2)) for m in 1:model.M]...)
    Ndivζ = vcat([fill(model.N[1][m] / ζ[m], K[m]) for m in 1:model.M]...)
    objective = MultiModalMuSig.λ_objective(λ, ∇λ, ν, Ndivζ, sumθ, μ, invΣ)
    @fact objective --> roughly(L)

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
    @fact ∇λ --> roughly(grad)

    model.μ = μ
    model.λ[1] = λ
    model.ν[1] = ν
    model.ζ[1] = ζ
    model.θ[1] = θ
    MultiModalMuSig.update_λ!(model, 1)
    @fact (model.λ[1] .> -20.0) | (model.λ[1] .< 20.0) --> fill(true, sum(K))
end

facts("ν_objective") do
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
    ∇ν = Array(Float64, sum(K))
    Ndivζ = vcat([fill(model.N[1][m] / ζ[m], K[m]) for m in 1:model.M]...)
    objective = MultiModalMuSig.ν_objective(ν, ∇ν, λ, Ndivζ, μ, invΣ)
    @fact objective --> roughly(L)

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
    @fact ∇ν --> roughly(grad)

    model.μ = μ
    model.λ[1] = λ
    model.ν[1] = ν
    model.ζ[1] = ζ
    MultiModalMuSig.update_ν!(model, 1)
    @fact (model.ν[1] .> 0.0) --> fill(true, sum(K))
    @fact (model.λ[1] .< 100.0) --> fill(true, sum(K))
end

facts("update_ζ!") do
    model = MultiModalMuSig.IMMCTM(K, α, features, X)
    model.λ = [[1,2,3,4,1], [2,3,1,4,2]]
    model.ν = [[1,1,1,2,1], [1,3,1,2,1]]
    MultiModalMuSig.update_ζ!(model, 1)

    ζ = Float64[exp(1.5) + exp(2.5), exp(3.5) + exp(5) + exp(1.5)]
    @fact model.ζ[1] --> roughly(ζ)
end

facts("update_θ!") do
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

    θ = Array(Float64, 2, 2)
    θ[1, 1] = exp(1 + digamma(0.1) - digamma(0.3) + digamma(0.1) - digamma(1.1))
    θ[2, 1] = exp(2 + digamma(0.1) - digamma(0.2) + digamma(1.0) - digamma(2.0))
    θ[1, 2] = exp(1 + digamma(0.1) - digamma(0.3) + digamma(1.0) - digamma(1.1))
    θ[2, 2] = exp(2 + digamma(0.1) - digamma(0.2) + digamma(1.0) - digamma(2.0))
    θ[:, 1] ./= sum(θ[:, 1])
    θ[:, 2] ./= sum(θ[:, 2])

    @fact sum(model.θ[1][1], 1) --> roughly(ones(size(X[1][1])[1])')
    @fact model.θ[1][1] --> roughly(θ)
    @fact any(model.θ[1][1] .< 0.0) --> false
end

facts("update_μ!") do
    model = MultiModalMuSig.IMMCTM(K, α, features, X)
    model.λ = [[1,2,3,4,1], [2,3,1,4,2]]

    MultiModalMuSig.update_μ!(model)

    @fact model.μ --> roughly([1.5, 2.5, 2.0, 4.0, 1.5])
end

facts("update_Σ!") do
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
    @fact model.Σ --> roughly(Σ)
    @fact model.invΣ --> roughly(inv(Σ))
end

facts("update_γ!") do
    model = MultiModalMuSig.IMMCTM(K, α, features, X)
    model.θ[1][1] = [0.4 0.1; 0.6 0.9]
    model.θ[2][1] = [0.3 0.5; 0.7 0.5]
    MultiModalMuSig.update_γ!(model)

    γ1 = [0.1 + 5 * 0.4 + 8 * 0.1, 0.1 + 4 * 0.3 + 9 * 0.5]
    γ2 = [0.1 + 5 * 0.4 + 4 * 0.3, 0.1 + 8 * 0.1 + 9 * 0.5]
    @fact model.γ[1][1][1] --> roughly(γ1)
    @fact model.γ[1][1][2] --> roughly(γ2)
end

facts("update_Elnϕ!") do
    model = MultiModalMuSig.IMMCTM(K, α, features, X)
    model.γ[1][1][1] .= [1, 2]

    MultiModalMuSig.update_Elnϕ!(model)

    @fact model.Elnϕ[1][1][1][1] --> roughly(digamma(1) - digamma(3))
end

facts("calculate_ElnPϕ") do
    model = MultiModalMuSig.IMMCTM(K, α, features, X)
    ElnPϕ = 0.0
    @pending @fact MultiModalMuSig.calculate_ElnPϕ(model) --> roughly(ElnPϕ)
end

facts("calculate_ElnPη") do
    model = MultiModalMuSig.IMMCTM(K, α, features, X)
    ElnPη = 0.0
    @pending @fact MultiModalMuSig.calculate_ElnPη(model) --> roughly(ElnPη)
end

facts("calculate_ElnPZ") do
    model = MultiModalMuSig.IMMCTM(K, α, features, X)
    ElnPZ = 0.0
    @pending @fact MultiModalMuSig.calculate_ElnPZ(model) --> roughly(ElnPZ)
end

facts("calculate_ElnPX") do
    model = MultiModalMuSig.IMMCTM(K, α, features, X)
    ElnPX = 0.0
    @pending @fact MultiModalMuSig.calculate_ElnPX(model) --> roughly(ElnPX)
end

facts("calculate_ElnQϕ") do
    model = MultiModalMuSig.IMMCTM(K, α, features, X)
    ElnQϕ = 0.0
    @pending @fact MultiModalMuSig.calculate_ElnQϕ(model) --> roughly(ElnQϕ)
end

facts("calculate_ElnQη") do
    model = MultiModalMuSig.IMMCTM(K, α, features, X)
    ElnQη = 0.0
    @pending @fact MultiModalMuSig.calculate_ElnQη(model) --> roughly(ElnQη)
end

facts("calculate_ElnQZ") do
    model = MultiModalMuSig.IMMCTM(K, α, features, X)
    ElnQZ = 0.0
    @pending @fact MultiModalMuSig.calculate_ElnQZ(model) --> roughly(ElnQZ)
end

facts("calculate_elbo") do
    model = MultiModalMuSig.IMMCTM(K, α, features, X)
    @fact MultiModalMuSig.calculate_elbo(model) --> less_than(0.0)
end

facts("fit") do
    model = MultiModalMuSig.IMMCTM(K, α, features, X)
    ll = MultiModalMuSig.fit!(model, maxiter=1, verbose=false) 
    @fact length(ll) --> 1
    @fact length(ll[1]) --> 2
end

facts("loglikelihoods") do 
    η = [[1.0, 2.0], [2.0, 3.0]]
    θ = [exp(η[d]) ./ sum(exp(η[d])) for d in 1:2]

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

    @fact res --> roughly(ll)
end

FactCheck.exitstatus()
end
