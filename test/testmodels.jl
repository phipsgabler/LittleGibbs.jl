using Distributions
using StatsBase: var

########## GDEMO ##################
# Example partially based on https://stats.stackexchange.com/a/266672/234110 and
# http://www2.stat.duke.edu/~rcs46/modern_bayes17/lecturesModernBayes17/lecture-7/07-gibbs.pdf.
# We follow the model used in https://github.com/TuringLang/Turing.jl/blob/60724e22a9831066fc2e0e82d428bd4922bda6e8/test/test_utils/models.jl#L2,
# with normally distributed observations and conjugate priors for both parameters:
# λ ~ Gamma(α₀, θ₀)
# m ~ Normal(0, √(1 / λ))
# Xᵢ ~ Normal(m, √(1 / λ)) (iid)
    

function gdemo_forward(N, α₀, θ₀)
    λ = rand(Gamma(α₀, θ₀))
    σ = √(1 / λ)
    m = rand(Normal(0, σ))
    x = rand(Normal(m, σ), N)
    return (m=m, λ=λ), x
end

function gdemo_logjoint(α₀, θ₀, x, m, λ)
    σ = √(1 / λ)
    return logpdf(Gamma(α₀, θ₀), λ) +
        logpdf(Normal(0, σ), m) +
        sum(logpdf.(Normal(m, σ), x))
end

function gdemo_statistics(x)
    # The conditionals and posterior can be formulated in terms of the following statistics:
    N = length(x) # number of samples
    x̄ = mean(x) # sample mean
    s² = var(x; mean=x̄, corrected=false) # sample variance
    return N, x̄, s²
end

function gdemo_cond_m(α₀, θ₀, x, λ)
    N, x̄, s² = gdemo_statistics(x)
    mₙ = N * x̄ / (N + 1)
    λₙ = λ * (N + 1)
    σₙ = √(1 / λₙ)
    return Normal(mₙ, σₙ)
end

function gdemo_cond_λ(α₀, θ₀, x, m)
    N, x̄, s² = gdemo_statistics(x)
    αₙ = α₀ + (N - 1) / 2
    βₙ = (s² * N / 2 + m^2 / 2 + inv(θ₀))
    return Gamma(αₙ, inv(βₙ))
end


function gdemo(x; α₀=2.0, θ₀=inv(3.0))
    cond_m((λ,)) = gdemo_cond_m(α₀, θ₀, x, λ)
    cond_λ((m,)) = gdemo_cond_λ(α₀, θ₀, x, m)
    conditionals = (m=cond_m, λ=cond_λ)
    return conditionals
end



########### 1D ISING MODEL #########################
function ising_conditional(β, i, M)
    return function (neighbours)
        x_prev = neighbours[mod1(i - 1, M - 1)]
        x_next = neighbours[mod1(i, M - 1)]
        p₁ = exp(-β * (x_prev + x_next))
        p₂ = exp(-β * (abs(x_prev - 1) + abs(x_next - 1)))
        return Bernoulli(p₁ / (p₁ + p₂))
    end
end

function isingdemo(β, M)
    # One dimensional Ising model on a circle, without external field;
    # see https://stats.stackexchange.com/a/312044/234110
    conditional_dists = [Symbol(:x, i) => ising_conditional(β, i, M) for i = 1:M]
    conditionals = merge(NamedTuple(), conditional_dists)
end

function ising_conditional(β, i, j, M, N)
    return function (neighbours)
        x_prev = neighbours[mod1(i - 1, M - 1)]
        x_next = neighbours[mod1(i, M - 1)]
        x_down = neighbours[mod1(j - 1, N - 1)]
        x_up = neighbours[mod1(j, N - 1)]
        p₁ = exp(-β * (x_prev + x_next + x_up + x_down))
        p₂ = exp(-β * (abs(x_prev - 1) + abs(x_next - 1) + abs(x_up - 1) + abs(x_down - 1)))
        return Bernoulli(p₁ / (p₁ + p₂))
    end
end

function isingdemo(β, M, N)
    # One dimensional Ising model on a circle, without external field;
    # see https://stats.stackexchange.com/a/312044/234110
    conditional_dists = [Symbol(:x, i, j) => ising_conditional(β, i, j, M, N) for i = 1:M, j = 1:N]
    conditionals = merge(NamedTuple(), conditional_dists)
end

function visualize_ising(θ::AbstractVector{<:Real})
    join(v == 1 ? '\u2588' : ' ' for v in values(θ))
end

function visualize_ising(θ::AbstractMatrix{<:Real})
    join((join(v == 1 ? '\u2588' : ' ' for v in row) for row in eachrow(θ)), '\n')
end
