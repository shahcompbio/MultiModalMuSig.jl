mutable struct ILDA
    K::Int          # topics
    D::Int          # documents
    I::Int          # features
    J::Vector{Int}  # values feature

    η::Vector{Float64}              # topic hyperparameter
    λ::Vector{Matrix{Float64}}      # topic variational parameters
    β::Vector{Matrix{Float64}}      # topic-feature distributions
    Elnβ::Vector{Matrix{Float64}}   # expectation of ln(β)
    
    α::Float64                  # doc-topic hyperparameter
    γ::Matrix{Float64}          # doc-topic variational parameters
    θ::Matrix{Float64}          # doc-topic distributions
    Elnθ::Matrix{Float64}       # expectation of ln(θ)
    ϕ::Vector{Matrix{Float64}}  # Z variational parameters

    features::Matrix{Int}
    X::Vector{Matrix{Int}}

    converged::Bool
    elbo::Float64
    ll::Float64

    function ILDA(k::Int, α::Float64, η::Vector{Float64},
            features::Matrix{Int}, X::Vector{Matrix{Int}})
        model = new()

        model.K = k
        model.α = α
        model.η = copy(η)
        model.X = X
        model.D = length(X)
        model.I = size(features, 2)
        model.J = vec(maximum(features, dims=1))
        model.features = features

        model.λ = [rand(1:100, model.J[i], model.K) for i in 1:model.I]
        model.Elnβ = [
            Array{Float64}(undef, model.J[i], model.K) for i in 1:model.I
        ]
        update_Elnβ!(model)

        model.γ = fill(1.0, model.K, model.D)
        model.Elnθ = Array{Float64}(undef, model.K, model.D)
        update_Elnθ!(model)

        model.ϕ = [
            fill(1.0 / model.K, model.K, size(model.X[d], 1))
            for d in 1:model.D
        ]

        model.converged = false

        return model
    end
end

function ILDA(k::Int, α::Float64, η::Float64, features::Matrix{Int},
        X::Vector{Matrix{Int}})
    I = size(features, 2)
    return ILDA(k, α, fill(η, I), features, X)
end

function update_ϕ!(model::ILDA)
    for d in 1:model.D
        for w in 1:size(model.X[d], 1)
            model.ϕ[d][:, w] .= model.Elnθ[:, d]

            v = model.X[d][w, 1]
            for i in 1:model.I
                j = model.features[v, i]
                model.ϕ[d][:, w] .+= model.Elnβ[i][j, :]
            end
        end

        model.ϕ[d] .= exp.(model.ϕ[d]) ./ sum(exp.(model.ϕ[d]), dims=1)
    end
end

function update_Elnθ!(model::ILDA)
    model.Elnθ .= digamma.(model.γ) .- digamma.(sum(model.γ, dims=1))
end

function update_γ!(model::ILDA)
    model.γ .= model.α

    for d in 1:model.D
        model.γ[:, d] .+= model.ϕ[d] * model.X[d][:, 2]
    end

    update_Elnθ!(model)
end

function update_θ!(model::ILDA)
    model.θ = model.γ ./ sum(model.γ, dims=1)
end

function update_Elnβ!(model::ILDA)
    for i in 1:model.I
        model.Elnβ[i] .= (
            digamma.(model.λ[i]) .- digamma.(sum(model.λ[i], dims=1))
        )
    end
end

function update_λ!(model::ILDA)
    for i in 1:model.I
        model.λ[i] .= model.η[i]
    end

    for d in 1:model.D
        Nϕ = model.ϕ[d]' .* model.X[d][:, 2]

        for w in 1:size(model.X[d], 1)
            v = model.X[d][w, 1]

            for i in 1:model.I
                j = model.features[v, i]
                model.λ[i][j, :] .+= Nϕ[w, :]
            end
        end
    end

    update_Elnβ!(model)
end

function update_β!(model::ILDA)
    model.β = [model.λ[i] ./ sum(model.λ[i], dims=1) for i in 1:model.I]
end

function calculate_ElnPβ(model::ILDA)
    lnp = 0.0
    for i in 1:model.I
        lnp += model.K * (
            lgamma(model.J[i] * model.η[i]) - model.J[i] * lgamma(model.η[i])
        )
        lnp += (model.η[i] - 1) * sum(model.Elnβ[i])
    end
    return lnp
end

function calculate_ElnPθ(model::ILDA)
    lnp = model.D * (lgamma(model.K * model.α) - model.K * lgamma(model.α))
    lnp += (model.α - 1) * sum(model.Elnθ)
    return lnp
end

function calculate_ElnPZ(model::ILDA)
    lnp = 0.0
    for d in 1:model.D
        lnp += sum(model.ϕ[d] .* model.Elnθ[:, d] .* model.X[d][:, 2]')
    end
    return lnp
end

function calculate_ElnPX(model::ILDA)
    lnp = 0.0
    for d in 1:model.D
        Nϕ = model.ϕ[d]' .* model.X[d][:, 2]

        for w in 1:size(model.X[d], 1)
            v = model.X[d][w, 1]

            for i in 1:model.I
                j = model.features[v, i]
                lnp += sum(Nϕ[w, :] .* model.Elnβ[i][j, :])
            end
        end
    end
    return lnp
end

function calculate_ElnQβ(model::ILDA)
    lnq = 0.0
    for i in 1:model.I
        lnq = sum(lgamma.(model.λ[i])) - sum(lgamma.(sum(model.λ[i], dims=1)))
        lnq -= sum((model.λ[i] .- 1) .* model.Elnβ[i])
    end
    return lnq
end

function calculate_ElnQθ(model::ILDA)
    lnq = sum(lgamma.(model.γ)) - sum(lgamma.(sum(model.γ, dims=1)))
    lnq -= sum((model.γ .- 1) .* model.Elnθ)
    return lnq
end

function calculate_ElnQZ(model::ILDA)
    lnq = 0.0
    for d in 1:model.D
        lnq += sum(log.(model.ϕ[d] .^ model.ϕ[d]))
    end
    return lnq
end

function calculate_elbo(model::ILDA)
    elbo = 0.0
    elbo += calculate_ElnPβ(model)
    elbo += calculate_ElnPθ(model)
    elbo += calculate_ElnPZ(model)
    elbo += calculate_ElnPX(model)
    elbo -= calculate_ElnQβ(model)
    elbo -= calculate_ElnQθ(model)
    elbo -= calculate_ElnQZ(model)
    return elbo
end

function calculate_loglikelihood(X::Vector{Matrix{Int}}, features::Matrix{Int}, 
                                 θ::Matrix{Float64}, β::Vector{Matrix{Float64}})
    ll = 0.0

    K = size(θ, 1)
    I = size(features, 2)
    N = 0
    for d in 1:length(X)
        N += sum(X[d][:, 2])

        for w in 1:size(X[d], 1)
            v = X[d][w, 1]

            pw = 0.0
            for k in 1:K
                tmp = θ[k, d]
                for i in 1:I
                    j = features[v, i]
                    tmp *= β[i][j, k]
                end
                pw += tmp
            end
            ll += X[d][w, 2] * log(pw)
        end
    end

    return ll / N
end

function calculate_loglikelihood(X::Vector{Matrix{Int}}, model::ILDA)
    return calculate_loglikelihood(X, model.features, model.θ, model.β)
end

function calculate_loglikelihood(model::ILDA)
    return calculate_loglikelihood(model.X, model.features, model.θ, model.β)
end

function fit!(model::ILDA; maxiter=1000, tol=1e-4, verbose=true)
    ll = Float64[]

    for iter in 1:maxiter
        update_γ!(model)
        update_ϕ!(model)
        update_λ!(model)

        update_β!(model)
        update_θ!(model)

        push!(ll, calculate_loglikelihood(model))

        if verbose
            println("$iter\tLog-likelihood: ", ll[end])
        end

        if length(ll) > 10 && check_convergence(ll, tol=tol)
            model.converged = true
            break
        end
    end
    model.elbo = calculate_elbo(model)
    model.ll = ll[end]

    return ll
end

function unsmoothed_update_ϕ!(model::ILDA)
    for d in 1:model.D
        for w in 1:size(model.X[d], 1)
            model.ϕ[d][:, w] .= exp.(model.Elnθ[:, d])

            v = model.X[d][w, 1]
            for i in 1:model.I
                j = model.features[v, i]
                model.ϕ[d][:, w] .*= model.β[i][j, :]
            end
        end

        model.ϕ[d] .= model.ϕ[d] ./ sum(model.ϕ[d], dims=1)
    end
end

function transform(model::ILDA, X::Vector{Matrix{Int}};
                   maxiter=1000, tol=1e-4, verbose=false)

    newmodel = LDA(model.K, model.α, model.η, X)
    newmodel.β = deepcopy(model.β)

    ll = Float64[]

    for iter in 1:maxiter
        update_γ!(newmodel)
        unsmoothed_update_ϕ!(newmodel)

        update_θ!(newmodel)

        push!(ll, calculate_loglikelihood(newmodel))

        if verbose
            println("$iter\tLog-likelihood: ", ll[end])
        end

        if length(ll) > 10 && check_convergence(ll, tol=tol)
            newmodel.converged = true
            break
        end
    end

    if !newmodel.converged
        warn("transform did not converge")
    end

    return newmodel.θ
end

function fit_heldout(Xheldout::Vector{Matrix{Int}}, model::ILDA;
        maxiter=100, verbose=false)

    heldout_model = ILDA(model.K, model.α, model.η, model.features, Xheldout)
    heldout_model.λ = deepcopy(model.λ)
    heldout_model.Elnβ = deepcopy(model.Elnβ)

    ll = Float64[]
    for iter in 1:maxiter
        update_γ!(heldout_model)
        update_ϕ!(heldout_model)

        push!(ll, calculate_loglikelihood(Xheldout, heldout_model))

        if verbose
            println("$iter\tLog-likelihood: ", ll[end])
        end

        if length(ll) > 10 && check_convergence(ll)
            heldout_model.converged = true
            break
        end
    end
    heldout_model.elbo = calculate_elbo(heldout_model)
    heldout_model.ll = ll[end]

    return heldout_model
end
