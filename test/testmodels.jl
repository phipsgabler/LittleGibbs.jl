using Distributions
using StatsBase: var

function gdemo(x::AbstractArray; α₀ = 2.0, β₀ = 3.0)
    # Example partially based on https://stats.stackexchange.com/a/266672/234110.
    # There, α₀ = β₀ = 1 is assumed.
    # We follow the model used in https://github.com/TuringLang/Turing.jl/blob/60724e22a9831066fc2e0e82d428bd4922bda6e8/test/test_utils/models.jl#L2,
    # with normally distributed observations and conjugate priors for both parameters:
    # λ ~ Gamma(α₀, 1 / β₀)
    # m ~ Normal(0, 1 / λ)
    # Xᵢ ~ Normal(m, 1 / λ) (iid)
    
    # The conditionals and posterior can be formulated in terms of the following statistics:
    N = length(x)            # number of samples
    x̄ = mean(x)              # sample mean
    s² = var(x; mean = x̄, corrected = false)    # sample variance

    # Gibbs conditionals
    m_given_λ((λ,)) = Normal(x̄, 1 / (N * λ))
    λ_given_m((m,)) = Gamma(N / 2, 2 / (N * s² + N * (m - x̄)^2))
    conditionals = (m = m_given_λ, λ = λ_given_m)

    # Exact posterior
    αₙ = α₀ + N / 2
    βₙ = β₀ + (N * s² + (N * x̄^2) / (1 + N)) / 2
    posterior_marginals = (m = LocationScale(0, βₙ / αₙ, TDist(2αₙ)), λ = Gamma(αₙ, 1 / βₙ))

    return conditionals, posterior_marginals
end


function ising_conditional(i, N, β)
    return function (neighbours)
        x_prev = neighbours[mod1(i - 1, N - 1)]
        x_next = neighbours[mod1(i, N - 1)]
        p₁ = exp(-β * (x_prev + x_next))
        p₂ = exp(-β * (abs(x_prev - 1) + abs(x_next - 1)))
        return Bernoulli(p₁ / (p₁ + p₂))
    end
end

function ising(N, β)
    # One dimensional Ising model on a circle, without external field;
    # see https://stats.stackexchange.com/a/312044/234110
    conditional_dists = [Symbol(:x, i) => ising_conditional(i, N, β) for i = 1:N]
    conditionals = merge(NamedTuple(), conditional_dists)
end
