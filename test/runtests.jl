using LittleGibbs
using Distributions
using Plots

function gdemo(x::AbstractArray)
    # Example partially based on https://stats.stackexchange.com/a/266672/234110.
    # There, α₀ = β₀ = 1 is assumed.
    # We follow the model used in https://github.com/TuringLang/Turing.jl/blob/60724e22a9831066fc2e0e82d428bd4922bda6e8/test/test_utils/models.jl#L2,
    # with normally distributed observations and conjugate priors for both parameters:
    # λ ~ Gamma(α₀, 1 / β₀)
    # m ~ Normal(0, 1 / λ)
    # Xᵢ ~ Normal(m, 1 / λ) (iid)

    α₀ = 2.0
    β₀ = 3.0
    
    # The conditionals and posterior can be formulated in terms of the following statistics:
    N = length(x)            # number of samples
    x̄ = mean(x)              # sample mean
    s² = var(x; mean = x̄)    # sample variance
    
    m_given_λ((λ,)) = Normal(x̄, 1 / (N * λ))
    λ_given_m((m,)) = Gamma(N / 2, 2 / ((N - 1) * s² + N * (m - x̄)^2))

    # Exact posterior
    function posterior(m, λ)
        αₙ = α₀ + N / 2
        βₙ = β₀ + (N * s² + (N * x̄^2) / (1 + N)) / 2
        mₙ = (N * x̄) / (1 + N)
        λₙ = 1 + N
        
        dgamma = Gamma(αₙ, 1 / βₙ)
        dnormal = Normal(mₙ, 1 / λₙ)
        
        return pdf(dgamma, λ) * pdf(dnormal, m)
        # return (m = LocationScale(0, βₙ / αₙ, TDist(2αₙ)),
                # λ = Gamma(αₙ, 1 / βₙ))
    end

    return (m = m_given_λ, λ = λ_given_m), posterior
end

function main()
    conditionals, exact_posterior = gdemo(sqrt(3) .* randn(30) .+ 15)
    model = BlockModel(conditionals)
    sampler = Gibbs((m = 0.0, λ = 1.0))
    chain = sample(model, sampler, 11_000)

    m̂ = mean(t -> t.params[:m], chain[1000:end])
    λ̂ = mean(t -> t.params[:λ], chain[1000:end])
    @show m̂, λ̂

    x = y = -2:0.05:2
    @show exact_posterior(-1.5, -1.5)
    # display(contour(x, y, exact_posterior))
end

main()
