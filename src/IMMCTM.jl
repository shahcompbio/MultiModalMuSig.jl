type IMMCTM
    K::Vector{Int}          # topics
    D::Int                  # documents
    N::Vector{Vector{Int}}  # observations per document modality
    M::Int                  # modalities
    I::Vector{Int}          # features per modality
    J::Vector{Vector{Int}}  # values per modality feature
    V::Vector{Int}          # vocab items per modality

    μ::Vector{Float64}
    Σ::Matrix{Float64}
    invΣ::Matrix{Float64}
    α::Vector{Float64}

    ζ::Vector{Vector{Float64}}
    θ::Vector{Vector{Matrix{Float64}}}
    λ::Vector{Vector{Float64}}
    ν::Vector{Vector{Float64}}
    γ::Vector{Vector{Vector{Vector{Float64}}}}
    Elnϕ::Vector{Vector{Vector{Vector{Float64}}}}

    features::Vector{Matrix{Int}}
    X::Vector{Vector{Matrix{Int}}}

    converged::Bool
    elbo::Float64
    perplexities::Vector{Float64}

    function IMMCTM(k::Vector{Int}, α::Vector{Float64},
                    features::Vector{Matrix{Int}},
                    X::Vector{Vector{Matrix{Int}}})
        model = new()

        model.K = k
        model.α = α
        model.features = features
        model.X = X

        model.D = length(X)
        model.M = length(features)
        model.I = [size(features[m])[2] for m in 1:model.M]
        model.J = [vec(maximum(features[m], 1)) for m in 1:model.M]
        model.V = [size(features[m])[1] for m in 1:model.M]
        model.N = [[sum(X[d][m][:, 2]) for m in 1:model.M] for d in 1:model.D]

        MK = sum(model.K)

        model.μ = zeros(MK)
        model.Σ = eye(MK)
        model.invΣ = eye(MK)

        model.θ = [
            [
               rand(Dirichlet(model.K[m], 1.0 / model.K[m]), size(X[d][m])[1])
                for m in 1:model.M
            ] for d in 1:model.D
        ]

        model.γ = [
            [
                [
                    Array(Float64, model.J[m][i]) for i in 1:model.I[m]
                ] for k in 1:model.K[m]
            ] for m in 1:model.M
        ]
        model.Elnϕ = deepcopy(model.γ)
        update_γ!(model)

        model.λ = [randn(MK) for d in 1:model.D]
        model.ν = [ones(MK) for d in 1:model.D]

        model.ζ = [Array(Float64, model.M) for d in 1:model.D]
        for d in 1:model.D update_ζ!(model, d) end

        model.converged = false

        return model
    end
end

function λ_objective(λ::Vector{Float64}, ∇λ::Vector{Float64},
        ν::Vector{Float64}, Ndivζ::Vector{Float64}, sumθ::Vector{Float64},
        μ::Vector{Float64}, invΣ::Matrix{Float64})

    diff = λ .- μ
    Eeη = exp(λ .+ 0.5ν)

    if length(∇λ) > 0
        ∇λ .= -invΣ * diff .+ sumθ .- Ndivζ .* Eeη
    end

    return -0.5 * (diff' * invΣ * diff)[1] + sum(λ .* sumθ) - sum(Ndivζ .* Eeη)
end

function calculate_sumθ(model::IMMCTM, d::Int)
    return vcat(
        [
            vec(sum(model.θ[d][m] .* model.X[d][m][:, 2]', 2))
            for m in 1:model.M
        ]...
    )
end

function calculate_Ndivζ(model::IMMCTM, d::Int)
    return vcat(
        [
            fill(model.N[d][m] / model.ζ[d][m], model.K[m]) for m in 1:model.M
        ]...
    )
end

function update_λ!(model::IMMCTM, d::Int)
    opt = Opt(:LD_MMA, sum(model.K))
    lower_bounds!(opt, -20.0)
    upper_bounds!(opt, 20.0)
    xtol_rel!(opt, 1e-4)
    xtol_abs!(opt, 1e-4)

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

function ν_objective(ν::Vector{Float64}, ∇ν::Vector{Float64},
        λ::Vector{Float64}, Ndivζ::Vector{Float64}, μ::Vector{Float64},
        invΣ::Matrix{Float64})

    Eeη = exp(λ .+ 0.5ν)

    if length(∇ν) > 0
        ∇ν .= -0.5diag(invΣ) .- (Ndivζ / 2) .* Eeη .+ (1 ./ (2ν))
    end

    return -0.5 * trace(diagm(ν) * invΣ) - sum(Ndivζ .* Eeη) + sum(log(ν)) / 2
end

function update_ν!(model::IMMCTM, d::Int)
    opt = Opt(:LD_MMA, sum(model.K))
    lower_bounds!(opt, 1e-10)
    upper_bounds!(opt, 100.0)
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

function update_ζ!(model::IMMCTM, d::Int)
    start = 1
    for m in 1:model.M
        stop = start + model.K[m] - 1
        model.ζ[d][m] = sum(
            exp(model.λ[d][start:stop] .+ 0.5 * model.ν[d][start:stop])
        )
        start += model.K[m]
    end
end

function update_θ!(model::IMMCTM, d::Int)
    for m in 1:model.M
        for w in 1:size(model.X[d][m])[1]
            v = model.X[d][m][w, 1]

            offset = 0
            for k in 1:model.K[m]
                model.θ[d][m][k, w] = exp(model.λ[d][offset + k])

                for i in 1:model.I[m]
                    model.θ[d][m][k, w] *= exp(
                        model.Elnϕ[m][k][i][model.features[m][v, i]]
                    )
                end
            end
            offset += model.K[m]

            model.θ[d][m][:, w] ./= sum(model.θ[d][m][:, w])
        end
    end
end

function update_μ!(model::IMMCTM)
    model.μ .= mean(model.λ)
end

function update_Σ!(model::IMMCTM)
    model.Σ .= sum(diagm.(model.ν))
    for d in 1:model.D
        diff = model.λ[d] .- model.μ
        model.Σ .+= diff * diff'
    end
    model.Σ ./= model.D
    model.invΣ .= inv(model.Σ)
end

function update_Elnϕ!(model::IMMCTM)
    for m in 1:model.M
        for k in 1:model.K[m]
            for i in 1:model.I[m]
                model.Elnϕ[m][k][i] .= digamma(model.γ[m][k][i]) .-
                    digamma(sum(model.γ[m][k][i]))
            end
        end
    end
end

function update_γ!(model::IMMCTM)
    for m in 1:model.M
        for k in 1:model.K[m]
            for i in 1:model.I[m]
                for j in 1:model.J[m][i]
                    model.γ[m][k][i][j] = model.α[m] 
                end
            end
        end
    end
    for d in 1:model.D
        for m in 1:model.M
            Nθ = model.θ[d][m] .* model.X[d][m][:, 2]'
            for w in 1:size(model.X[d][m])[1]
                v = model.X[d][m][w, 1]
                for k in 1:model.K[m]
                    for i in 1:model.I[m]
                        model.γ[m][k][i][model.features[m][v, i]] += Nθ[k, w]
                    end
                end
            end
        end
    end
    update_Elnϕ!(model)
end

function logmvbeta(vals)
    r = 0.0
    for v in vals
        r += lgamma(v)
    end
    r -= lgamma(sum(vals))

    return r
end

function calculate_ElnPϕ(model::IMMCTM)
    lnp = 0.0

    for m in 1:model.M
        for k in 1:model.K[m]
            for i in 1:model.I[m]
                lnp -= logmvbeta(fill(model.α[m], model.J[m][i]))
                for j in 1:model.J[m][i]
                    lnp += (model.α[m] - 1) * model.Elnϕ[m][k][i][j]
                end
            end
        end
    end

    return lnp
end

function calculate_ElnPη(model::IMMCTM)
    lnp = 0.0

    for d in 1:model.D
        diff = model.λ[d] .- model.μ
        lnp += 0.5 * (
            logdet(model.invΣ) -
            sum(model.K) * log(2π) -
            trace(diagm(model.ν[d]) * model.invΣ) -
            (diff' * model.invΣ * diff)[1]
        )
    end

    return lnp
end

function calculate_ElnPZ(model::IMMCTM)
    lnp = 0.0

    for d in 1:model.D
        Eeη = exp(model.λ[d] .+ 0.5model.ν[d])
        sumθ = calculate_sumθ(model, d)
        Ndivζ = calculate_Ndivζ(model, d)

        lnp += sum(model.λ[d] .* sumθ)
        lnp -= sum(Ndivζ .* Eeη) - sum(model.N[d])
        lnp -= sum(model.N[d] .* log(model.ζ[d]))
    end

    return lnp
end

function calculate_ElnPX(model::IMMCTM)
    lnp = 0.0

    for d in 1:model.D
        for m in 1:model.M
            for w in 1:size(model.X[d][m])[1]
                v = model.X[d][m][w, 1]
                for i in 1:model.I[m]
                    for k in 1:model.K[m]
                        lnp += model.X[d][m][w, 2] * model.θ[d][m][k, w] *
                            model.Elnϕ[m][k][i][model.features[m][v, i]]
                    end
                end
            end
        end
    end

    return lnp
end

function calculate_ElnQϕ(model::IMMCTM)
    lnq = 0.0

    for m in 1:model.M
        for k in 1:model.K[m]
            for i in 1:model.I[m]
                lnq += -logmvbeta(model.γ[m][k][i])
                for j in 1:model.J[m][i]
                    lnq += (model.γ[m][k][i][j] - 1) * model.Elnϕ[m][k][i][j]
                end
            end
        end
    end
    return lnq
end

function calculate_ElnQη(model::IMMCTM)
    lnq = 0.0
    for d in 1:model.D
        lnq += -0.5 * (sum(log(model.ν[d])) + sum(model.K) * (log(2π) + 1))
    end
    return lnq
end

function calculate_ElnQZ(model::IMMCTM)
    lnq = 0.0
    for d in 1:model.D
        for m in 1:model.M
            lnq += sum(model.X[d][m][:, 2]' .* log(model.θ[d][m] .^ model.θ[d][m]))
        end
    end
    return lnq
end

function calculate_elbo(model::IMMCTM)
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

function calculate_perplexity(X::Vector{Matrix{Int}},
        η::Vector{Vector{Float64}}, γ::Vector{Vector{Vector{Float64}}},
        features::Matrix{Int})

    K = length(η[1])
    D = length(X)
    I = size(features)[2]

    props = [exp(η[d]) ./ sum(exp(η[d])) for d in 1:D]
    ϕ = [[γ[k][i] ./ sum(γ[k][i]) for i in 1:I] for k in 1:K]

    sum_ll = 0.0
    N = 0
    for d in 1:D
        for w in 1:size(X[d])[1]
            v = X[d][w, 1]
            pw = 0.0
            for k in 1:K
                tmp = props[d][k]
                for i in 1:I
                    tmp *= ϕ[k][i][features[v, i]]
                end
                pw += tmp
            end
            sum_ll += X[d][w, 2] * log(pw)
            N += X[d][w, 2]
        end
    end

    return exp(-sum_ll / N)
end

function calculate_perplexities(X::Vector{Vector{Matrix{Int}}}, model::IMMCTM)
    perps = Array(Float64, model.M)

    offset = 1
    for m in 1:model.M
        mk = offset:(offset + model.K[m] - 1)
        η = [model.λ[d][mk] for d in 1:model.D]
        Xm = [X[d][m] for d in 1:model.D]

        perps[m] = calculate_perplexity(Xm, η, model.γ[m], model.features[m])

        offset += model.K[m]
    end

    return perps
end

function check_convergence(perps::Vector{Vector{Float64}})
    return maximum(abs(perps[end - 1] .- perps[end]) ./ perps[end]) < 1e-4
end

function fitdoc!(model::IMMCTM, d::Int)
    update_ζ!(model, d)
    update_θ!(model, d)
    update_ν!(model, d)
    update_λ!(model, d)
end

function fit!(model::IMMCTM; maxiter=100, verbose=true)
    perps = Vector{Float64}[]
    for iter in 1:maxiter
        for d in 1:model.D
            fitdoc!(model, d)
        end

        update_μ!(model)
        update_Σ!(model)
        update_γ!(model)

        push!(perps, calculate_perplexities(model.X, model))

        if verbose
            println("$iter\tPerplexities: ", join(perps[end], ", "))
        end

        if length(perps) > 10 && check_convergence(perps)
            model.converged = true
            break
        end
    end
    model.elbo = calculate_elbo(model)
    model.perplexities = perps[end]

    return perps
end

function calculate_heldout_perplexities(Xhidden::Vector{Vector{Matrix{Int}}},
        Xobs::Vector{Vector{Matrix{Int}}}, model::IMMCTM;
        maxiter=100, verbose=false)

    test_model = IMMCTM(model.K, model.α, model.features, Xobs)
    test_model.μ .= model.μ
    test_model.Σ .= model.Σ
    test_model.invΣ .= model.invΣ
    test_model.γ .= model.γ
    test_model.Elnϕ .= model.Elnϕ

    perps = Vector{Float64}[]
    for iter in 1:maxiter
        for d in 1:test_model.D
            fitdoc!(model, d)
        end

        push!(perps, calculate_perplexities(Xobs, test_model))

        if verbose
            println("$iter\tPerplexities: ", join(perps[end], ", "))
        end

        if length(perps) > 10 && check_convergence(perps)
            test_model.converged = true
            break
        end
    end

    if !test_model.converged
        warn("test model not converged")
    end

    return calculate_perplexities(Xhidden, test_model)
end
