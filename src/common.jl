function logmvbeta(vals)
    r = 0.0
    for v in vals
        r += lgamma(v)
    end
    r -= lgamma(sum(vals))

    return r
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

function ν_objective(ν::Vector{Float64}, ∇ν::Vector{Float64},
        λ::Vector{Float64}, Ndivζ::Vector{Float64}, μ::Vector{Float64},
        invΣ::Matrix{Float64})

    Eeη = exp(λ .+ 0.5ν)

    if length(∇ν) > 0
        ∇ν .= -0.5diag(invΣ) .- (Ndivζ / 2) .* Eeη .+ (1 ./ (2ν))
    end

    return -0.5 * trace(diagm(ν) * invΣ) - sum(Ndivζ .* Eeη) + sum(log(ν)) / 2
end

function calculate_logprops(λ::Vector{Float64}, Ks::Vector{Int})
    logprops = Array(Float64, length(λ))
    start = 1
    for k in Ks
        stop = start + k - 1
        logprops[start:stop] .= λ[start:stop] - logsumexp(λ[start:stop]) 
        start = stop + 1
    end
    return logprops
end

function calculate_f(λ::Vector{Float64}, Etz::Vector{Float64},
        μ::Vector{Float64}, invΣ::Matrix{Float64}, Ks::Vector{Int})
    diff = λ .- μ
    logprops = calculate_logprops(λ, Ks)
    return (logprops' * Etz - 0.5 * diff' * invΣ * diff)[1]
end

function calculate_∇f(λ::Vector{Float64}, Etz::Vector{Float64},
        sum_Etz::Vector{Float64}, μ::Vector{Float64}, invΣ::Matrix{Float64},
        Ks::Vector{Int})
    diff = λ .- μ
    props = exp(calculate_logprops(λ, Ks))
    return Etz .- props .* sum_Etz .- invΣ * diff
end

function calculate_∇2f(λ::Vector{Float64}, Etz::Vector{Float64},
        sum_Etz::Vector{Float64}, μ::Vector{Float64}, invΣ::Matrix{Float64},
        Ks::Vector{Int})
    diff = λ .- μ
    props = exp(calculate_logprops(λ, Ks))
    return (-diagm(props) .+ props * props') .* sum_Etz .- invΣ
end

function laplace_λ_objective(λ::Vector{Float64}, ∇λ::Vector{Float64},
        Etz::Vector{Float64}, sum_Etz::Vector{Float64}, μ::Vector{Float64},
        invΣ::Matrix{Float64}, Ks::Vector{Int})
    if length(∇λ) > 0
        ∇λ .= calculate_∇f(λ, Etz, sum_Etz, μ, invΣ, Ks)
    end
    return calculate_f(λ, Etz, μ, invΣ, Ks)
end

function check_convergence(metric::Vector{Vector{Float64}}; tol=1e-4)
    reldiff = maximum(abs(metric[end - 1] .- metric[end]) ./ abs(metric[end]))
    return reldiff < tol
end

