function metafit!(model::IMMCTM; restarts=10, maxiter=100, verbose=true)
    topicpoints = [
        Array(Float64, sum(model.J[m]), model.K[m] * restarts)
        for m in 1:model.M
    ]

    for r in 1:restarts
        println("restart $r")
        restart_model = deepcopy(model)
        fit!(restart_model, maxiter=maxiter, verbose=false)
        if !restart_model.converged
            warn("restart model not converged")
        end

        for m in 1:model.M
            topic = Array(Float64, sum(model.J[m]))
            for k in 1:model.K[m]
                topicpoints[m][:, r + k - 1] .= vcat(restart_model.γ[m][k]...)
            end
        end
    end

    centres = [kmeans(topicpoints[m], model.K[m], init=:rand).centers for m in 1:model.M]
    for m in 1:model.M
        for k in 1:model.K[m]
            start = 1
            for i in 1:model.I[m]
                stop = start + model.J[m][i] - 1

                model.γ[m][k][i] .= centres[m][start:stop, k]
                nan_mask = isnan(model.γ[m][k][i])
                replace = topicpoints[m][start:stop, k]
                model.γ[m][k][i][nan_mask] .= replace[nan_mask]

                start = stop + 1
            end
        end
    end
    update_Elnϕ!(model)

    perps = Vector{Float64}[]
    for iter in 1:maxiter
        for d in 1:model.D
            fitdoc!(model, d)
        end

        update_μ!(model)
        update_Σ!(model)
        update_γ!(model)

        push!(perps, calculate_perplexities(model))

        if verbose
            println("Iteration: $iter\tPerplexities: ", join(perps[end], ", "))
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
