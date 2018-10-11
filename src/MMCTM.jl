mutable struct MMCTM
    K::Vector{Int}          # topics
    D::Int                  # documents
    N::Vector{Vector{Int}}  # observations per document modality
    M::Int                  # modalities
    V::Vector{Int}          # vocab items per modality

    μ::Vector{Float64}                      # doc-topic mean
    Σ::Matrix{Float64}                      # doc-topic covariance
    invΣ::Matrix{Float64}                   # inverse Σ
    props::Vector{Vector{Vector{Float64}}}  # doc-topic
    α::Vector{Float64}                      # topic-word hyperparameter
    ϕ::Vector{Vector{Vector{Float64}}}      # topic-word

    # variational parameters
    ζ::Vector{Vector{Float64}}              # doc-topic normalizer
    θ::Vector{Vector{Matrix{Float64}}}      # Z
    λ::Vector{Vector{Float64}}              # doc-topic mean
    ν::Vector{Vector{Float64}}              # doc-topic variance
    γ::Vector{Vector{Vector{Float64}}}      # topic-word
    Elnϕ::Vector{Vector{Vector{Float64}}}   # expectation of ln(ϕ)

    X::Vector{Vector{Matrix{Int}}}

    converged::Bool
    elbo::Float64
    ll::Vector{Float64}

    function MMCTM(k::Vector{Int}, α::Vector{Float64}, V::Vector{Int},
            X::Vector{Vector{Matrix{Int}}}; init=:random)
        model = new()

        model.K = copy(k)
        model.α = copy(α)
        model.X = X
        model.D = length(X)
        model.M = length(k)
        model.N = [[sum(X[d][m][:, 2]) for m in 1:model.M] for d in 1:model.D]

        model.V = copy(V)

        MK = sum(model.K)

        model.μ = zeros(MK)
        model.Σ = Matrix{Float64}(I, MK, MK)
        model.invΣ = Matrix{Float64}(I, MK, MK)
        model.props = [
            [Array{Float64}(undef, model.K[m]) for m in 1:model.M]
            for d in 1:model.D
        ]

        model.θ = [
            [
                fill(1.0 / model.K[m], model.K[m], size(model.X[d][m])[1])
                for m in 1:model.M
            ] for d in 1:model.D
        ]

        if init == :random
            model.γ = [
                [rand(1:100, model.V[m]) for k in 1:model.K[m]]
                for m in 1:model.M
            ]
        elseif init == :document
            model.γ = [
                [ones(model.V[m]) for k in 1:model.K[m]] for m in 1:model.M
            ]
            for m in 1:model.M
                seed_docs = sample(1:model.D, model.K[m], replace=false)
                for k in model.K[m]
                    d = seed_docs[k]
                    model.γ[m][k][X[d][m][:, 1]] .+= X[d][m][:, 2]
                end
            end
        else
            error("init must be either :random or :document")
        end
        model.Elnϕ = deepcopy(model.γ)
        update_Elnϕ!(model)
        model.ϕ = deepcopy(model.γ)

        model.λ = [zeros(MK) for d in 1:model.D]
        model.ν = [ones(MK) for d in 1:model.D]

        model.ζ = [Array{Float64}(undef, model.M) for d in 1:model.D]
        for d in 1:model.D update_ζ!(model, d) end

        model.converged = false

        return model
    end
end

function MMCTM(k::Vector{Int}, α::Vector{Float64},
        X::Vector{Vector{Matrix{Int}}})
    D = length(X)
    M = length(k)
    V = zeros(Int, M)
    for d in 1:D
        for m in 1:M
            if size(X[d][m], 1) > 0
                V[m] = max(V[m], maximum(X[d][m][:, 1]))
            end
        end
    end

    return MMCTM(k, α, V, X)
end

function calculate_sumθ(model::MMCTM, d::Int)
    return vcat(
        [
            vec(sum(model.θ[d][m] .* model.X[d][m][:, 2]', dims=2))
            for m in 1:model.M
        ]...
    )
end

function calculate_Ndivζ(model::MMCTM, d::Int)
    return vcat(
        [
            fill(model.N[d][m] / model.ζ[d][m], model.K[m]) for m in 1:model.M
        ]...
    )
end

function update_λ!(model::MMCTM, d::Int)
    opt = Opt(:LD_MMA, sum(model.K))
    xtol_rel!(opt, 1e-4)
    #xtol_abs!(opt, 1e-4)

    Ndivζ = calculate_Ndivζ(model, d)
    sumθ = calculate_sumθ(model, d)

    max_objective!(
        opt,
        (λ, ∇λ) -> λ_objective(
            λ, ∇λ, model.ν[d], Ndivζ, sumθ, model.μ, model.invΣ
        )
    )
    (optobj, optλ, ret) = optimize(opt, model.λ[d])
    model.λ[d] .= optλ
end

function update_props!(model::MMCTM)
    for d in 1:model.D
        offset = 1
        for m in 1:model.M
            η = model.λ[d][offset:(offset + model.K[m] - 1)]
            model.props[d][m] .= exp.(η) ./ sum(exp.(η))
            offset += model.K[m]
        end
    end
end

function update_ν!(model::MMCTM, d::Int)
    opt = Opt(:LD_MMA, sum(model.K))
    lower_bounds!(opt, 1e-7)
    xtol_rel!(opt, 1e-4)
    xtol_abs!(opt, 1e-4)

    Ndivζ = calculate_Ndivζ(model, d)

    max_objective!(
        opt,
        (ν, ∇ν) -> ν_objective(ν, ∇ν, model.λ[d], Ndivζ, model.μ, model.invΣ)
    )
    (optobj, optν, ret) = optimize(opt, model.ν[d])
    model.ν[d] .= optν
end

function update_ζ!(model::MMCTM, d::Int)
    start = 1
    for m in 1:model.M
        stop = start + model.K[m] - 1
        model.ζ[d][m] = sum(
            exp.(model.λ[d][start:stop] .+ 0.5 * model.ν[d][start:stop])
        )
        start += model.K[m]
    end
end

function update_θ!(model::MMCTM, d::Int)
    offset = 0
    for m in 1:model.M
        for w in 1:size(model.X[d][m], 1)
            v = model.X[d][m][w, 1]

            for k in 1:model.K[m]
                model.θ[d][m][k, w] = exp(
                    model.λ[d][offset + k] + model.Elnϕ[m][k][v]
                )
            end
        end
        model.θ[d][m] ./= sum(model.θ[d][m], dims=1)
        offset += model.K[m]
    end
end

function update_μ!(model::MMCTM)
    model.μ .= mean(model.λ)
end

function update_Σ!(model::MMCTM)
    model.Σ .= sum(diagm.(0 .=> model.ν))
    for d in 1:model.D
        diff = model.λ[d] .- model.μ
        model.Σ .+= diff * diff'
    end
    model.Σ ./= model.D
    model.invΣ .= inv(model.Σ)
end

function update_Elnϕ!(model::MMCTM)
    for m in 1:model.M
        for k in 1:model.K[m]
            model.Elnϕ[m][k] .= (
                digamma.(model.γ[m][k]) .- digamma(sum(model.γ[m][k]))
            )
        end
    end
end

function update_γ!(model::MMCTM)
    for m in 1:model.M
        for k in 1:model.K[m]
            model.γ[m][k] .= model.α[m] 
        end
    end
    for d in 1:model.D
        for m in 1:model.M
            Nθ = model.θ[d][m] .* model.X[d][m][:, 2]'
            for w in 1:size(model.X[d][m])[1]
                v = model.X[d][m][w, 1]
                for k in 1:model.K[m]
                    model.γ[m][k][v] += Nθ[k, w]
                end
            end
        end
    end
    update_Elnϕ!(model)
end

function update_ϕ!(model::MMCTM)
    for m in 1:model.M
        for k in 1:model.K[m]
            model.ϕ[m][k] .= model.γ[m][k] ./ sum(model.γ[m][k])
        end
    end
end

function update_α!(model::MMCTM)
    opt = Opt(:LD_MMA, 1)
    lower_bounds!(opt, 1e-7)
    xtol_rel!(opt, 1e-5)
    xtol_abs!(opt, 1e-5)

    for m in 1:model.M
        sum_Elnϕ = sum(sum(model.Elnϕ[m][k] for k in 1:model.K[m]))

        max_objective!(
            opt,
            (α, ∇α) -> α_objective(α, ∇α, sum_Elnϕ, model.K[m], model.V[m])
        )

        (optobj, optα, ret) = optimize(opt, model.α[m:m])
        model.α[m] = optα[1]
    end
end

function calculate_ElnPϕ(model::MMCTM)
    lnp = 0.0

    for m in 1:model.M
        for k in 1:model.K[m]
            lnp -= logmvbeta(fill(model.α[m], model.V[m]))
            for v in 1:model.V[m]
                lnp += (model.α[m] - 1) * model.Elnϕ[m][k][v]
            end
        end
    end

    return lnp
end

function calculate_ElnPη(model::MMCTM)
    lnp = 0.0

    for d in 1:model.D
        diff = model.λ[d] .- model.μ
        lnp += 0.5 * (
            logdet(model.invΣ) -
            sum(model.K) * log(2π) -
            tr(diagm(0 => model.ν[d]) * model.invΣ) -
            (diff' * model.invΣ * diff)[1]
        )
    end

    return lnp
end

function calculate_ElnPZ(model::MMCTM)
    lnp = 0.0

    for d in 1:model.D
        Eeη = exp.(model.λ[d] .+ 0.5model.ν[d])
        sumθ = calculate_sumθ(model, d)
        Ndivζ = calculate_Ndivζ(model, d)

        lnp += sum(model.λ[d] .* sumθ)
        lnp -= sum(Ndivζ .* Eeη) - sum(model.N[d])
        lnp -= sum(model.N[d] .* log.(model.ζ[d]))
    end

    return lnp
end

function calculate_ElnPX(model::MMCTM)
    lnp = 0.0

    for d in 1:model.D
        for m in 1:model.M
            for w in 1:size(model.X[d][m])[1]
                v = model.X[d][m][w, 1]
                for k in 1:model.K[m]
                    lnp += (
                        model.X[d][m][w, 2] * model.θ[d][m][k, w] *
                        model.Elnϕ[m][k][v]
                    )
                end
            end
        end
    end

    return lnp
end

function calculate_ElnQϕ(model::MMCTM)
    lnq = 0.0

    for m in 1:model.M
        for k in 1:model.K[m]
            lnq += -logmvbeta(model.γ[m][k])
            for v in 1:model.V[m]
                lnq += (model.γ[m][k][v] - 1) * model.Elnϕ[m][k][v]
            end
        end
    end
    return lnq
end

function calculate_ElnQη(model::MMCTM)
    lnq = 0.0
    for d in 1:model.D
        lnq += -0.5 * (sum(log.(model.ν[d])) + sum(model.K) * (log(2π) + 1))
    end
    return lnq
end

function calculate_ElnQZ(model::MMCTM)
    lnq = 0.0
    for d in 1:model.D
        for m in 1:model.M
            lnq += sum(
                model.X[d][m][:, 2]' .* log.(model.θ[d][m] .^ model.θ[d][m])
            )
        end
    end
    return lnq
end

function calculate_elbo(model::MMCTM)
    elbo = 0.0
    elbo += calculate_ElnPϕ(model)
    elbo += calculate_ElnPη(model)
    elbo += calculate_ElnPZ(model)
    elbo += calculate_ElnPX(model)
    elbo -= calculate_ElnQϕ(model)
    elbo -= calculate_ElnQη(model)
    elbo -= calculate_ElnQZ(model)
    return elbo
end

function calculate_docmodality_loglikelihood(X::Matrix{Int},
        props::Vector{Float64}, ϕ::Vector{Vector{Float64}})

    K = length(ϕ)

    ll = 0.0
    for w in 1:size(X, 1)
        v = X[w, 1]
        pw = 0.0
        for k in 1:K
            pw += props[k] * ϕ[k][v]
        end
        ll += X[w, 2] * log(pw)
    end

    return ll / sum(X[:, 2])
end

function calculate_modality_loglikelihood(X::Vector{Matrix{Int}},
        props::Vector{Vector{Float64}}, ϕ::Vector{Vector{Float64}})
    D = length(X)

    ll = 0.0
    N = 0
    for d in 1:D
        doc_N = sum(X[d][:, 2])
        if doc_N > 0
            doc_ll = calculate_docmodality_loglikelihood(X[d], props[d], ϕ)
            ll += doc_ll * doc_N
            N += doc_N
        end
    end

    return ll / N
end

function calculate_loglikelihoods(X::Vector{Vector{Matrix{Int}}},
                                  props::Vector{Vector{Vector{Float64}}},
                                  ϕ::Vector{Vector{Vector{Float64}}})
    D = length(X)
    M = length(ϕ)
    K = [length(ϕ[m]) for m in 1:M]

    ll = Array{Float64}(undef, M)

    offset = 1
    for m in 1:M
        Xm = [X[d][m] for d in 1:D]
        propsm = [props[d][m] for d in 1:D]

        ll[m] = calculate_modality_loglikelihood(Xm, propsm, ϕ[m])

        offset += K[m]
    end

    return ll
end

function calculate_loglikelihoods(X::Vector{Vector{Matrix{Int}}}, model::MMCTM)
    return calculate_loglikelihoods(X, model.props, model.ϕ)
end

function calculate_loglikelihoods(model::MMCTM)
    return calculate_loglikelihoods(model.X, model.props, model.ϕ)
end

function fitdoc!(model::MMCTM, d::Int)
    update_ζ!(model, d)
    update_θ!(model, d)
    update_ν!(model, d)
    update_λ!(model, d)
end

function fit!(model::MMCTM; maxiter=100, tol=1e-4, verbose=true, autoα=false)
    ll = Vector{Float64}[]

    for iter in 1:maxiter
        for d in 1:model.D
            fitdoc!(model, d)
        end

        update_μ!(model)
        update_Σ!(model)
        update_γ!(model)
        if autoα
            update_α!(model)
        end

        update_props!(model)
        update_ϕ!(model)

        push!(ll, calculate_loglikelihoods(model))

        if verbose
            println("$iter\tLog-likelihoods: ", join(ll[end], ", "))
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

function unsmoothed_update_θ!(model::MMCTM, d::Int)
    offset = 0
    for m in 1:model.M
        for w in 1:size(model.X[d][m], 1)
            v = model.X[d][m][w, 1]

            for k in 1:model.K[m]
                model.θ[d][m][k, w] = exp(model.λ[d][offset + k]) * model.ϕ[m][k][v]
            end
        end
        model.θ[d][m] ./= sum(model.θ[d][m], dims=1)
        offset += model.K[m]
    end
end

function transform(model::MMCTM, X::Vector{Vector{Matrix{Int}}};
                   maxiter=1000, tol=1e4, verbose=false)

    newmodel = MMCTM(model.K, model.α, X)
    newmodel.ϕ = deepcopy(model.ϕ)

    ll = Vector{Float64}[]
    for iter in 1:maxiter
        for d in 1:newmodel.D
            update_ζ!(newmodel, d)
            unsmoothed_update_θ!(newmodel, d)
            update_ν!(newmodel, d)
            update_λ!(newmodel, d)
        end

        update_μ!(newmodel)
        update_Σ!(newmodel)

        update_props!(newmodel)

        push!(ll, calculate_loglikelihoods(newmodel))

        if verbose
            println("$iter\tLog-likelihoods: ", join(ll[end], ", "))
        end

        if length(ll) > 10 && check_convergence(ll, tol=tol)
            newmodel.converged = true
            break
        end
    end

    return newmodel
end

function fit_heldout(Xheldout::Vector{Vector{Matrix{Int}}}, model::MMCTM;
        maxiter=100, verbose=false)

    heldout_model = MMCTM(model.K, model.α, Xheldout)
    heldout_model.μ .= model.μ
    heldout_model.Σ .= model.Σ
    heldout_model.invΣ .= model.invΣ
    heldout_model.γ = deepcopy(model.γ)
    heldout_model.Elnϕ = deepcopy(model.Elnϕ)
    heldout_model.ϕ = deepcopy(model.ϕ)

    ll = Vector{Float64}[]
    for iter in 1:maxiter
        for d in 1:heldout_model.D
            fitdoc!(heldout_model, d)
        end

        update_props!(model)

        push!(ll, calculate_loglikelihoods(heldout_model))

        if verbose
            println("$iter\tLog-likelihoods: ", join(ll[end], ", "))
        end

        if length(ll) > 10 && check_convergence(ll)
            heldout_model.converged = true
            break
        end
    end

    return heldout_model
end

function predict_modality_η(Xobs::Vector{Vector{Matrix{Int}}}, m::Int,
        model::MMCTM; maxiter=100, verbose=false)
    obsM = setdiff(1:model.M, m)

    moffset = sum(model.K[1:(m - 1)])
    unobsMK = (moffset + 1):(moffset + model.K[m])
    obsMK = setdiff(1:sum(model.K), unobsMK)

    obsmodel = MMCTM(model.K[obsM], model.α[obsM], Xobs)
    obsmodel.μ .= model.μ[obsMK]
    obsmodel.Σ .= model.Σ[obsMK, obsMK]
    obsmodel.invΣ .= model.invΣ[obsMK, obsMK]
    obsmodel.γ = deepcopy(model.γ[obsM])
    obsmodel.Elnϕ = deepcopy(model.Elnϕ[obsM])

    ll = Vector{Float64}[]
    for iter in 1:maxiter
        for d in 1:obsmodel.D
            fitdoc!(obsmodel, d)
        end

        push!(ll, calculate_loglikelihoods(Xobs, obsmodel))

        if verbose
            println("$iter\tLog-likelihoods: ", join(ll[end], ", "))
        end

        if length(ll) > 10 && check_convergence(ll)
            obsmodel.converged = true
            break
        end
    end

    if !obsmodel.converged
        warn("model not converged.")
    end

    η = [
        (
            model.μ[unobsMK] .+ model.Σ[unobsMK, obsMK] *
            model.invΣ[obsMK, obsMK] * (obsmodel.λ[d] .- model.μ[obsMK])
        )
        for d in 1:obsmodel.D
    ]

    return η
end
