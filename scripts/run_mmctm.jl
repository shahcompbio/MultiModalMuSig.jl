using ArgParse
using CSV
using DataFrames
using DelimitedFiles
using Distributed
using JLD
using LinearAlgebra
@everywhere using MultiModalMuSig
@everywhere using NLopt
using ProgressMeter
@everywhere using Random
using Statistics
using StatsBase

function getargs()
    s = ArgParseSettings()

    @add_arg_table! s begin
        # inputs
        "counts"
            help = "mutation counts tsv files"
            nargs = '+'
            required = true
        "-k", "--num-sigs"
            help = "number of signatures for each mutation type"
            arg_type = Int
            nargs = '+'
            dest_name = "k"
            required = true
        "-m", "--modality-labels"
            help = "modality labels for output"
            arg_type = String
            nargs = '+'
            dest_name = "modalities"
            required = true
        # outputs
        "--model"
            help = "model output jld file"
        "--mean"
            help = "gaussian mean output file"
        "--cov"
            help = "guassian covariance matrix output tsv file"
        "--cor"
            help = "correlation output tsv file"
        "--sigs"
            help = "signatures output tsv file"
        "--props"
            help = "signature proportions output tsv file"
        # options
        "--restarts", "-r"
            help = "number of restarts for each stage of fitting"
            arg_type = Int
            default = 1000
        "--verbose", "-v"
            help = "print output"
            action = :store_true
        "--progress", "-p"
            help = "print output"
            action = :store_true
        "--seed", "-s"
            help = "random state seed"
            arg_type = Int
            default = 147959412
        "--alpha", "-a"
            help = "topic dirichlet hyperparameter value"
            arg_type = Float64
            default = 0.1
    end

    return parse_args(s)
end

function readtsv(filename)
    return CSV.read(filename, DataFrame, delim='\t')
end

@everywhere function fit_restart(seed, K, α, V, counts)
    Random.seed!(seed)
    NLopt.srand(seed)

    model = MultiModalMuSig.MMCTM(K, α, V, counts)
    MultiModalMuSig.fit!(model, maxiter=1000, tol=1e-4, verbose=false)
    return model
end

function pick_optimal_modality_models(models)
    M = length(models[1].ll)

    ll = Array{Float64}(undef, length(models), M)
    for i in 1:length(models)
        ll[i, :] .= models[i].ll
    end

    opt_models = vec([x[1] for x in findmax(ll; dims=1)[2]])

    return models[opt_models]
end

function fit_seed_models(counts, K, α, V, seeds; progress=false)
    if progress
        models = @sync @showprogress pmap(
            (i) -> fit_restart(seeds[i], K, α, V, counts), 1:length(seeds)
        )
    else
        models = @sync pmap(
            (i) -> fit_restart(seeds[i], K, α, V, counts), 1:length(seeds)
        )
    end

    opt_models = pick_optimal_modality_models(models)
end

@everywhere function seed_and_fit_restart(seed, opt_models; verbose=false)
    Random.seed!(seed)
    NLopt.srand(seed)

    K = opt_models[1].K
    α = opt_models[1].α
    V = opt_models[1].V
    counts = opt_models[1].X

    model = MultiModalMuSig.MMCTM(K, α, V, counts)

    M = length(K)
    for m in 1:M
        model.γ[m] = deepcopy(opt_models[m].γ[m])
        model.Elnϕ[m] = deepcopy(opt_models[m].Elnϕ[m])
        model.ϕ[m] = deepcopy(opt_models[m].ϕ[m])
    end

    MultiModalMuSig.fit!(model, maxiter=1000, tol=1e-5, verbose=verbose)

    return model
end

function pick_optimal_model(models)
    ll = Array{Float64}(undef, length(models), models[1].M)
    for i in 1:length(models)
        ll[i, :] .= models[i].ll
    end

    ranks = Array{Float64}(undef, size(ll))
    for i in 1:size(ll, 2)
        ranks[:, i] .= denserank(abs.(ll[:, i]))
    end
    return models[findmin(mean(ranks, dims=2))[2]]
end

function seed_and_fit_model(opt_models, seeds; progress=false)
    if progress
        models = @sync @showprogress pmap(
            (i) -> seed_and_fit_restart(seeds[i], opt_models), 1:length(seeds)
        )
    else
        models = @sync pmap(
            (i) -> seed_and_fit_restart(seeds[i], opt_models), 1:length(seeds)
        )
    end

    return pick_optimal_model(models)
end

function fit_model(counts, K, α, V, restarts, verbose, seed, progress)
    Random.seed!(seed)
    seeds = rand(1:typemax(Int), restarts)

    seed_models = fit_seed_models(counts, K, α, V, seeds; progress=progress)
    if verbose
        println("Modality optimal model log-likelihoods:")
        for m in 1:length(K)
            println(m, ": ", seed_models[m].ll)
        end
    end

    model = seed_and_fit_model(seed_models, seeds; progress=progress)
    if verbose
        println("Seeded model log-likelihoods:")
        println(model.ll)
    end

    return model
end

function cov2cor(C::AbstractMatrix)
    sigma = sqrt.(diag(C))
    return C ./ (sigma*sigma')
end

function topicdf(model, terms, modalities)
    ms = String[]
    ks = Int[]
    vs = Int[]
    ts = String[]
    ps = Float64[]

    for m in 1:model.M
        for k in 1:model.K[m]
            probs = model.γ[m][k] ./ sum(model.γ[m][k])
            for v in 1:model.V[m]
                push!(ms, modalities[m])
                push!(ks, k)
                push!(vs, v)
                push!(ts, terms[m][v])
                push!(ps, probs[v])
            end
        end
    end
    return DataFrame(modality=ms, topic=ks, value=vs, term=ts, probability=ps)
end

function writesigs(filename, model, terms, modalities)
    sigs = topicdf(model, terms, modalities)
    CSV.write(filename, sigs, delim='\t')
end

function propdf(model, samples, modalities)
    props = Array{Float64}(undef, sum(model.K), length(samples))

    for d in 1:model.D
        start = 1
        stop = 0
        for m in 1:model.M
            stop = start + model.K[m] - 1

            expλ = exp.(model.λ[d][start:stop])
            props[start:stop, d] .= expλ ./ sum(expλ)

            start += model.K[m]
        end
    end

    props = DataFrame(props, :auto)
    rename!(props, samples)

    topiclabels = vcat(
        [["$(modalities[m])-$k" for k in 1:model.K[m]] for m in 1:model.M]...
    )

    return hcat(DataFrame(topic=topiclabels), props)
end

function writeprops(filename, model, samples, modalities)
    props = propdf(model, samples, modalities)
    CSV.write(filename, props, delim='\t')
end

function main()
    argv = getargs()

    if length(argv["counts"]) != length(argv["k"])
        error("Number of count files must match the number of K values.")
    end
    if length(argv["modalities"]) != length(argv["k"])
        error("Number of modality labels must match the number of K values.")
    end

    countdfs = [readtsv(f) for f in argv["counts"]]
    samples = [c for c in propertynames(countdfs[1]) if c != :term]
    counts = MultiModalMuSig.format_counts_mmctm(countdfs, samples)

    α = fill(argv["alpha"], length(argv["k"]))
    V = Int[size(c, 1) for c in countdfs]
    model = fit_model(
        counts, argv["k"], α, V, argv["restarts"], argv["verbose"],
        argv["seed"], argv["progress"]
    )

    if argv["verbose"]
        println("Log-likelihoods: $(model.ll)")
    end

    if argv["model"] !== nothing
        @save argv["model"] model
    end
    if argv["mean"] !== nothing
        writedlm(argv["mean"], model.μ)
    end
    if argv["cov"] !== nothing
        writedlm(argv["cov"], model.Σ)
    end
    if argv["cor"] !== nothing
        writedlm(argv["cor"], cov2cor(model.Σ))
    end
    if argv["sigs"] !== nothing
        terms = [df[!, :term] for df in countdfs]
        writesigs(argv["sigs"], model, terms, argv["modalities"])
    end
    if argv["props"] !== nothing
        writeprops(argv["props"], model, samples, argv["modalities"])
    end
end

main()
