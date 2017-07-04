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

function α_objective(α::Vector{Float64}, ∇α::Vector{Float64},
        sum_Elnϕ::Float64, K::Int, V::Int)

    if length(∇α) > 0
        ∇α[1] = K * V * (digamma(V * α[1]) - digamma(α[1])) + sum_Elnϕ
    end

    return K * (lgamma(V * α[1]) - V * lgamma(α[1])) + α[1] * sum_Elnϕ
end

function check_convergence(metric::Vector{Vector{Float64}}; tol=1e-4)
    reldiff = maximum(abs(metric[end - 1] .- metric[end]) ./ abs(metric[end]))
    return reldiff < tol
end

function check_convergence(metric::Vector{Float64}; tol=1e-4)
    reldiff = maximum(abs(metric[end - 1] - metric[end]) / abs(metric[end]))
    return reldiff < tol
end
