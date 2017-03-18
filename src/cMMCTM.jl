type cMMCTM
    K::Vector{Int}          # topics
    D::Int                  # documents
    N::Vector{Vector{Int}}  # observations per document modality
    M::Int                  # modalities
    V::Vector{Int}          # vocab items per modality

    μ::Vector{Float64}
    Σ::Matrix{Float64}
    invΣ::Matrix{Float64}
    α::Vector{Float64}

    ζ::Vector{Vector{Float64}}
    θ::Vector{Vector{Matrix{Float64}}}
    λ::Vector{Vector{Float64}}
    ν::Vector{Vector{Float64}}
    ENvk::Vector{Matrix{Float64}}
    VarNvk::Vector{Matrix{Float64}}
    ENk::Vector{Vector{Float64}}
    VarNk::Vector{Vector{Float64}}

    X::Vector{Vector{Matrix{Int}}}

    converged::Bool
    elbo::Float64
    ll::Vector{Float64}

    function cMMCTM(k::Vector{Int}, α::Vector{Float64}, V::Vector{Int},
            X::Vector{Vector{Matrix{Int}}})
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
        model.Σ = eye(MK)
        model.invΣ = eye(MK)

        model.θ = [
            [
               rand(Dirichlet(model.K[m], 1.0 / model.K[m]), size(X[d][m])[1])
                for m in 1:model.M
            ] for d in 1:model.D
        ]

        model.ENvk = [zeros(model.V[m], model.K[m]) for m in 1:model.M]
        model.VarNvk = deepcopy(model.ENvk)
        for d in 1:model.D
            for m in 1:model.M
                v = X[d][m][:, 1]
                counts = X[d][m][:, 2]
                model.ENvk[m][v, :] += counts .* model.θ[d][m]'
                model.VarNvk[m][v, :] += counts .* (
                    model.θ[d][m] .* (1 - model.θ[d][m])
                )'
            end
        end
        model.ENk = [vec(sum(model.ENvk[m], 1)) for m in 1:model.M]
        model.VarNk = [vec(sum(model.VarNvk[m], 1)) for m in 1:model.M]

        model.λ = [zeros(MK) for d in 1:model.D]
        model.ν = [ones(MK) for d in 1:model.D]

        model.ζ = [Array(Float64, model.M) for d in 1:model.D]
        for d in 1:model.D update_ζ!(model, d) end

        model.converged = false

        return model
    end
end

function cMMCTM(k::Vector{Int}, α::Vector{Float64},
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

    return cMMCTM(k, α, V, X)
end

function calculate_sumθ(model::cMMCTM, d::Int)
    return vcat(
        [
            vec(sum(model.θ[d][m] .* model.X[d][m][:, 2]', 2))
            for m in 1:model.M
        ]...
    )
end

function calculate_Ndivζ(model::cMMCTM, d::Int)
    return vcat(
        [
            fill(model.N[d][m] / model.ζ[d][m], model.K[m]) for m in 1:model.M
        ]...
    )
end

function update_λ!(model::cMMCTM, d::Int)
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

function update_ν!(model::cMMCTM, d::Int)
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

function update_ζ!(model::cMMCTM, d::Int)
    start = 1
    for m in 1:model.M
        stop = start + model.K[m] - 1
        model.ζ[d][m] = sum(
            exp(model.λ[d][start:stop] .+ 0.5 * model.ν[d][start:stop])
        )
        start += model.K[m]
    end
end

function update_θ!(model::cMMCTM, d::Int)
    offset = 0

    for m in 1:model.M
        ENvk = Array(Float64, model.K[m])
        VarNvk = Array(Float64, model.K[m])

        for w in 1:size(model.X[d][m], 1)
            v = model.X[d][m][w, 1]
            count = model.X[d][m][w, 2]

            @inbounds ENvk .= vec(count * model.θ[d][m][:, w])
            @inbounds VarNvk .= vec(
                count * model.θ[d][m][:, w] .* (1 - model.θ[d][m][:, w])
            )
            @inbounds model.ENvk[m][v, :] .-= ENvk
            @inbounds model.VarNvk[m][v, :] .-= VarNvk
            @inbounds model.ENk[m] .-= ENvk
            @inbounds model.VarNk[m] .-= VarNvk

            for k in 1:model.K[m]
                tmp1 = model.α[m] + model.ENvk[m][v, k]

                tmp2 = model.V[m] * model.α[m] + model.ENk[m][k]

                tmp3 = model.VarNvk[m][v, k] / (
                       2 * (model.α[m] + model.ENvk[m][v, k]) ^ 2
                )

                tmp4 = model.VarNk[m][k] / (
                     2 * (model.V[m] * model.α[m] + model.ENk[m][k]) ^ 2
                )

                model.θ[d][m][k, w] = tmp1 / tmp2 * exp(
                    model.λ[d][offset + k] - tmp3 + tmp4
                )
            end
            #model.θ[d][m][:, w] .= (
                #(model.α[m] + model.En[m][v, :]) ./ 
                #vec(model.V[m] * model.α[m] + sum(model.En[m], 1)) .* 
                #exp(
                    #model.λ[d][(offset + 1):(offset + model.K[m])] .-
                    #model.Varn[m][v, :] ./ (
                       #2 * (model.α[m] + model.En[m][v, :]) .^ 2
                    #) .+
                    #vec(sum(model.Varn[m], 1) ./ (
                        #2 * (model.V[m] * model.α[m] + sum(model.En[m], 1)) .^ 2
                    #))
                #)
            #)

            model.θ[d][m][:, w] ./= sum(model.θ[d][m][:, w])

            @inbounds ENvk .= vec(count * model.θ[d][m][:, w])
            @inbounds VarNvk .= vec(
                count * model.θ[d][m][:, w] .* (1 - model.θ[d][m][:, w])
            )
            @inbounds model.ENvk[m][v, :] .+= ENvk
            @inbounds model.VarNvk[m][v, :] .+= VarNvk
            @inbounds model.ENk[m] .+= ENvk
            @inbounds model.VarNk[m] .+= VarNvk
        end
        offset += model.K[m]
    end
end

function update_μ!(model::cMMCTM)
    model.μ .= mean(model.λ)
end

function update_Σ!(model::cMMCTM)
    model.Σ .= sum(diagm.(model.ν))
    for d in 1:model.D
        diff = model.λ[d] .- model.μ
        model.Σ .+= diff * diff'
    end
    model.Σ ./= model.D
    model.invΣ .= inv(model.Σ)
end

function update_α!(model::cMMCTM)
    opt = Opt(:LD_LBFGS, 1)
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

function calculate_ElnPϕ(model::cMMCTM)
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

function calculate_ElnPη(model::cMMCTM)
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

function calculate_ElnPZ(model::cMMCTM)
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

function calculate_ElnPX(model::cMMCTM)
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

function calculate_ElnQϕ(model::cMMCTM)
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

function calculate_ElnQη(model::cMMCTM)
    lnq = 0.0
    for d in 1:model.D
        lnq += -0.5 * (sum(log(model.ν[d])) + sum(model.K) * (log(2π) + 1))
    end
    return lnq
end

function calculate_ElnQZ(model::cMMCTM)
    lnq = 0.0
    for d in 1:model.D
        for m in 1:model.M
            lnq += sum(model.X[d][m][:, 2]' .* log(model.θ[d][m] .^ model.θ[d][m]))
        end
    end
    return lnq
end

function calculate_elbo(model::cMMCTM)
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
        η::Vector{Float64}, ϕ::Matrix{Float64})
    props = exp(η) ./ sum(exp(η))

    K = length(η)

    ll = 0.0
    for w in 1:size(X, 1)
        v = X[w, 1]
        pw = 0.0
        for k in 1:K
            pw += props[k] * ϕ[v, k]
        end
        ll += X[w, 2] * log(pw)
    end

    return ll / sum(X[:, 2])
end

function calculate_modality_loglikelihood(X::Vector{Matrix{Int}},
        η::Vector{Vector{Float64}}, ϕ::Matrix{Float64})
    D = length(X)

    ll = 0.0
    N = 0
    for d in 1:D
        doc_N = sum(X[d][:, 2])
        if doc_N > 0
            doc_ll = calculate_docmodality_loglikelihood(X[d], η[d], ϕ)
            ll += doc_ll * doc_N
            N += doc_N
        end
    end

    return ll / N
end

function calculate_loglikelihoods(X::Vector{Vector{Matrix{Int}}}, model::cMMCTM)
    ll = Array(Float64, model.M)

    offset = 1
    for m in 1:model.M
        mk = offset:(offset + model.K[m] - 1)
        η = [model.λ[d][mk] for d in 1:model.D]
        Xm = [X[d][m] for d in 1:model.D]
        ϕ = (
            (model.α[m] + model.ENvk[m]) ./
            (model.V[m] * model.α[m] + model.ENk[m])'
        )

        ll[m] = calculate_modality_loglikelihood(Xm, η, ϕ)

        offset += model.K[m]
    end

    return ll
end

function fitdoc!(model::cMMCTM, d::Int)
    update_ζ!(model, d)
    update_θ!(model, d)
    update_ν!(model, d)
    update_λ!(model, d)
end

function fit!(model::cMMCTM; maxiter=100, tol=1e-4, verbose=true, autoα=false)
    ll = Vector{Float64}[]

    for iter in 1:maxiter
        for d in 1:model.D
            fitdoc!(model, d)
        end

        update_μ!(model)
        update_Σ!(model)
        if autoα
            update_α!(model)
        end

        push!(ll, calculate_loglikelihoods(model.X, model))

        if verbose
            println("$iter\tLog-likelihoods: ", join(ll[end], ", "))
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

#function metafit(ks::Vector{Int}, α::Vector{Float64},
        #X::Vector{Vector{Matrix{Int}}}; restarts::Int=25)
    
    #M = length(ks)
    #nvals = [
        #maximum(maximum(X[d][m][:, 1]) for d in 1:model.D)
        #for m in 1:model.M
    #]
    #γs = Matrix{Float64}[
        #Array(Float64, nvals[m], restarts * ks[m]) for m in 1:M
    #]

    #for r in 1:restarts
        #println("restart $r")
        #model = cMMCTM(ks, α, X)
        #fit!(model, maxiter=1000, verbose=false)

        #for m in 1:M
            #for k in ks[m]
                #γs[m][:, r + k - 1] .= log(model.γ[m][k])
            #end
        #end
    #end

    #best_cost = fill(Inf, M)
    #best_centres = [Array(Float64, nvals[m], ks[m]) for m in 1:M]
    #res = [kmeans(γs[m], ks[m]) for m in 1:M]
    #for m in 1:M
        #for _ in 1:10
            #res = kmeans(γs[m], ks[m])
            #if res.totalcost < best_cost[m]
                #best_cost[m] = res.totalcost
                #best_centres[m] .= res.centers
            #end
        #end
    #end

    #model = cMMCTM(ks, α, X)
    #for m in 1:model.M
        #for k in 1:ks[m]
            #model.γ[m][k] = vec(exp(best_centres[m][:, k]))
        #end
    #end

    #fit!(model, maxiter=1000, verbose=false)
    #return model
#end

function fit_heldout(Xheldout::Vector{Vector{Matrix{Int}}}, model::cMMCTM;
        maxiter=100, verbose=false)

    heldout_model = cMMCTM(model.K, model.α, Xheldout)
    heldout_model.μ .= model.μ
    heldout_model.Σ .= model.Σ
    heldout_model.invΣ .= model.invΣ
    heldout_model.ENvk = deepcopy(model.ENvk)
    heldout_model.VarNvk = deepcopy(model.VarNvk)
    heldout_model.ENk = deepcopy(model.ENk)
    heldout_model.VarNk = deepcopy(model.VarNk)

    ll = Vector{Float64}[]
    for iter in 1:maxiter
        for d in 1:heldout_model.D
            fitdoc!(heldout_model, d)
        end

        push!(ll, calculate_loglikelihoods(Xheldout, heldout_model))

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
        model::cMMCTM; maxiter=100, verbose=false)
    obsM = setdiff(1:model.M, m)

    moffset = sum(model.K[1:(m - 1)])
    unobsMK = (moffset + 1):(moffset + model.K[m])
    obsMK = setdiff(1:sum(model.K), unobsMK)

    obsmodel = cMMCTM(model.K[obsM], model.α[obsM], Xobs)
    obsmodel.μ .= model.μ[obsMK]
    obsmodel.Σ .= model.Σ[obsMK, obsMK]
    obsmodel.invΣ .= model.invΣ[obsMK, obsMK]
    obsmodel.γ = deepcopy(model.γ[obsM])

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
