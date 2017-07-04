type LDA
    K::Int          # topics
    D::Int          # documents
    N::Vector{Int}  # observations per document modality
    V::Int          # vocab items per modality

    α::Float64              # topic hyperparameter
    λ::Matrix{Float64}      # topic variational parameters
    Elnλ::Matrix{Float64}
    
    η::Float64                  # doc-topic hyperparameter
    γ::Matrix{Float64}          # doc-topic variational parameters
    Elnγ::Matrix{Float64}
    ϕ::Vector{Matrix{Float64}}  # Z variational parameters

    X::Vector{Matrix{Int}}

    converged::Bool
    elbo::Float64
    ll::Float64

    function LDA(k::Int, α::Float64, η::Float64, V::Int,
            X::Vector{Matrix{Int}})
        model = new()

        model.K = k
        model.α = α
        model.η = η
        model.X = X
        model.D = length(X)
        model.N = [sum(X[d][:, 2]) for d in 1:model.D]
        model.V = V

        model.λ = α .+ rand(model.V, model.K)
        model.Elnλ = Array(Float64, model.V, model.K)
        update_Elnλ!(model)

        model.γ = 1.0 .+ fill(η, model.K, model.D)
        model.Elnγ = Array(Float64, model.K, model.D)
        update_Elnγ!(model)

        model.ϕ = [
            fill(1.0 / model.K, model.K, size(model.X[d], 1))
            for d in 1:model.D
        ]

        model.converged = false

        return model
    end
end

function LDA(k::Int, α::Float64, η::Float64, X::Vector{Matrix{Int}})
    D = length(X)
    V = 0
    for d in 1:D
        if size(X[d], 1) > 0
            V = max(V, maximum(X[d][:, 1]))
        end
    end

    return LDA(k, α, η, V, X)
end

function update_ϕ!(model::LDA)
    for d in 1:model.D
        model.ϕ[d] .= exp(
            model.Elnγ[:, d] .+ model.Elnλ[model.X[d][:, 1], :]'
        )
        model.ϕ[d] ./= sum(model.ϕ[d], 1)
    end
end

function update_Elnγ!(model::LDA)
    model.Elnγ .= digamma(model.γ) .- digamma(sum(model.γ, 1))
end

function update_γ!(model::LDA)
    model.γ .= model.α

    for d in 1:model.D
        model.γ[:, d] .+= model.ϕ[d] * model.X[d][:, 2]
    end

    update_Elnγ!(model)
end

function update_Elnλ!(model::LDA)
    model.Elnλ .= digamma(model.λ) .- digamma(sum(model.λ, 1))
end

function update_λ!(model::LDA)
    model.λ .= model.η

    for d in 1:model.D
        model.λ[model.X[d][:, 1], :] .+= model.ϕ[d]' .* model.X[d][:, 2]
    end

    update_Elnλ!(model)
end

function calculate_ElnPϕ(model::LDA)
    lnp = 0.0

    return lnp
end

function calculate_ElnPZ(model::LDA)
    lnp = 0.0

    return lnp
end

function calculate_ElnPX(model::LDA)
    lnp = 0.0

    return lnp
end

function calculate_ElnQϕ(model::LDA)
    lnq = 0.0

    return lnq
end

function calculate_ElnQZ(model::LDA)
    lnq = 0.0

    return lnq
end

# TODO
function calculate_elbo(model::LDA)
    elbo = 0.0
    elbo += calculate_ElnPϕ(model)
    elbo += calculate_ElnPZ(model)
    elbo += calculate_ElnPX(model)
    elbo -= calculate_ElnQϕ(model)
    elbo -= calculate_ElnQZ(model)
    return elbo
end

function calculate_loglikelihood(X::Vector{Matrix{Int}}, model::LDA)
    ll = 0.0

    β = model.λ ./ sum(model.λ, 1)
    θ = model.γ ./ sum(model.γ, 1)

    N = 0
    for d in 1:length(X)
        N += sum(X[d][:, 2])
        for w in 1:size(X[d], 1)
            v = X[d][w, 1]
            ll += X[d][w, 2] * log(dot(θ[:, d], β[v, :]))
        end
    end

    return ll / N
end

function fit!(model::LDA; maxiter=100, tol=1e-4, verbose=true, autoα=false)
    ll = Float64[]

    for iter in 1:maxiter
        update_γ!(model)
        update_ϕ!(model)
        update_λ!(model)

        push!(ll, calculate_loglikelihood(model.X, model))

        if verbose
            println("$iter\tLog-likelihood: ", ll[end])
        end

        if length(ll) > 10 && check_convergence(ll, tol=tol)
            model.converged = true
            break
        end
    end
    #model.elbo = calculate_elbo(model)
    model.ll = ll[end]

    return ll
end

function fit_heldout(Xheldout::Vector{Matrix{Int}}, model::LDA;
        maxiter=100, verbose=false)

    heldout_model = LDA(model.K, model.α, model.η, Xheldout)
    heldout_model.λ = deepcopy(model.λ)
    heldout_model.Elnλ = deepcopy(model.Elnλ)

    ll = Float64[]
    for iter in 1:maxiter
        update_γ!(model)
        update_ϕ!(model)

        push!(ll, calculate_loglikelihood(Xheldout, heldout_model))

        if verbose
            println("$iter\tLog-likelihood: ", ll[end])
        end

        if length(ll) > 10 && check_convergence(ll)
            heldout_model.converged = true
            break
        end
    end

    return heldout_model
end
