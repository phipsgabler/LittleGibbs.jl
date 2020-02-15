import HypothesisTests
using StatsBase: autocov
using Distributions: Normal

struct JointDistributionTest <: HypothesisTest
    statistic::Float64
end

function JointDistributionTest(
    g, θ::AbstractArray, y::AbstractArray, θ̃::AbstractArray, ỹ::AbstractArray
)
    M₁ = length(θ)
    @assert length(y) == M₁
    M₂ = length(θ̃)
    @assert length(ỹ) == M₂

    g₁ = g.(θ, y)
    g₂ = g.(θ̃, ỹ)
    ḡ₁ = mean(g)
    ḡ₂ = mean(g̃)
    σ̂² = var(g₁; mean = ḡ₁)
    τ̂² = mcvar(g₂)
    
    statistic = (ḡ₁ - ḡ₂) / (σ̂² / M₁ + τ̂² / M₂)
    return JointDistributionTest(statistic)
end

pvalue(test::JointDistributionTest) = pvalue(Normal(), test.statistic)


function mcvar(x::Vector{T}) where {T<:Real}
    # https://github.com/TuringLang/MCMCChains.jl/blob/9ab389648728064c85c8d331c1efbabc38649107/src/mcse.jl
    n = length(x)
    m = div(n - 2, 2)
    x_ = map(Float64, x)
    ghat = autocov(x_, [0, 1])
    Ghat = sum(ghat)
    value = -ghat[1] + 2 * Ghat
    for i in 1:m
        Ghat = min(Ghat, sum(autocov(x_, [2 * i, 2 * i + 1])))
        Ghat > 0 || break
        value += 2 * Ghat
    end
    return value / n
end


