using LittleGibbs
using Distributions


# Example from https://stats.stackexchange.com/a/266672/234110:
# Y ~ Normal(μ, 1/τ) with resp. priors

const N = 30    # number of samples
const ȳ = 15.0  # sample mean
const s² = 3.0  # sample variance

μ_given_τ((τ,)) = Normal(ȳ, 1 / (N * τ))
τ_given_μ((μ,)) = Gamma(N / 2, 2 / ((N - 1) * s² + N * (μ - ȳ)^2))

function main()
    println("started...")
    
    model = BlockModel((μ = μ_given_τ, τ = τ_given_μ))
    sampler = Gibbs((μ = 0.0, τ = 1.0))
    chain = sample(model, sampler, 1)
    show(chain)
end

main()
