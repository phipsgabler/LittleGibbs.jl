using Test
using HypothesisTests
using LittleGibbs

include("./testmodels.jl")


function test_gdemo_m(N, α₀, θ₀)
    # Testing consistency of conditional distribution with joint distribution,
    # see: Grosse, R. B. & Duvenaud, D. K. Testing MCMC Code. arXiv:1412.5218 [cs, stat] (2014).
    # p(m | λ, x)/p(m′ | λ, x) =!= p(m, λ, x)/p(m′, λ, x)
    
    (m, λ), x = gdemo_forward(N, α₀, θ₀)
    cond_m = gdemo_cond_m(α₀, θ₀, x, λ)
    m′ = m + randn()
    @test isapprox(logpdf(cond_m, m′) - logpdf(cond_m, m),
                   gdemo_logjoint(α₀, θ₀, x, m′, λ) - gdemo_logjoint(α₀, θ₀, x, m, λ),
                   rtol=0.1)
end

function test_gdemo_λ(N, α₀, θ₀)
    # Testing consistency of conditional distribution with joint distribution,
    # see: Grosse, R. B. & Duvenaud, D. K. Testing MCMC Code. arXiv:1412.5218 [cs, stat] (2014).
    # p(λ | m, x)/p(λ′ | m, x) =!= p(λ, m, x)/p(λ′, m, x)
    
    (m, λ), x = gdemo_forward(N, α₀, θ₀)
    cond_λ = gdemo_cond_λ(α₀, θ₀, x, m)
    λ′ = max(0, λ + randn())
    @test isapprox(logpdf(cond_λ, λ′) - logpdf(cond_λ, λ),
                   gdemo_logjoint(α₀, θ₀, x, m, λ′) - gdemo_logjoint(α₀, θ₀, x, m, λ),
                   rtol=0.2)
end

function test_gdemo(observations, α₀, θ₀, m_true, λ_true; m_init=0.0, λ_init=1.0)
    # Unit tests for conditionals
    test_gdemo_m(100, α₀, θ₀)
    test_gdemo_λ(100, α₀, θ₀)

    # Tests for expectations
    conditionals = gdemo(observations; α₀=α₀, θ₀=θ₀)
    
    model = BlockModel((m=conditionals.m, λ=(m,) -> Normal(λ_true, 0)))
    sampler = Gibbs((m=m_init, λ=λ_init))
    chain = sample(model, sampler, 10_000)
    chain_m = map(t -> t.params.m, chain[1_000:end])
    @test isapprox(mean(chain_m), m_true, atol=0.2, rtol=0.0)

    model = BlockModel((m=(λ,) -> Normal(m_true, 0), λ=conditionals.λ))
    sampler = Gibbs((m=m_init, λ=λ_init))
    chain = sample(model, sampler, 10_000)
    chain_λ = map(t -> t.params.λ, chain[1_000:end])
    @test isapprox(mean(chain_λ), λ_true, atol=0.2, rtol=0.0)

    model = BlockModel(conditionals)
    sampler = Gibbs((m=m_init, λ=λ_init))
    chain = sample(model, sampler, 10_000)
    chain_m = map(t -> t.params.m, chain[1_000:end])
    chain_λ = map(t -> t.params.λ, chain[1_000:end])
    @test isapprox(mean(chain_m), m_true, atol=0.2, rtol=0.0)
    @test isapprox(mean(chain_λ), λ_true, atol=0.2, rtol=0.0)
end


@testset "gdemo" begin
    N = 1000
    
    α₀, θ₀ = 2.0, inv(3.0)
    λ_true = rand(Gamma(α₀, θ₀))
    σ_true = √(1 / λ_true)
    m_true = rand(Normal(0, σ_true))
    test_gdemo(rand(Normal(m_true, σ_true), N), α₀, θ₀, m_true, λ_true; λ_init = α₀ * θ₀)
    
    α₀, θ₀ = 1.5, 1.5
    λ_true = rand(Gamma(α₀, θ₀))
    σ_true = √(1 / λ_true)
    m_true = rand(Normal(0, σ_true))
    test_gdemo(rand(Normal(m_true, σ_true), N), α₀, θ₀, m_true, λ_true; λ_init = α₀ * θ₀)
    
    # x = y = -2:0.05:2
    # @show exact_posterior(-1.5, -1.5)
    # display(contour(x, y, exact_posterior))
end

@testset "mixture" begin
    π = [0.5, 0.5]
    K = length(π)
    m = 0.5
    λ = 2.0
    σ = 0.1
    observations = [σ .* randn(10); 1 .+ σ .* randn(10)]
    N = length(observations)
    
    conditionals = mixture(π, K, m, λ, σ, observations)
    model = BlockModel(conditionals)
    sampler = Gibbs((z=rand((0, 1), N), μ=randn(2)))
    
    chain = sample(model, sampler, 11_000)
    chain_z = map(t -> t.params[:z], chain[1001:end])
    chain_μ = map(t -> t.params[:μ], chain[1001:end])
    
    ẑ = mean(chain_z)
    μ̂ = mean(chain_μ)
    z_true = [fill(1, 10); fill(2, 10)]
    μ_true = [0, 1]
    @info (ẑ=ẑ, μ̂=μ̂)
    @test isapprox(ẑ, z_true, atol=0.2, rtol=0.0) || isapprox(ẑ, z_true[[11:20; 1:10]], atol=0.2, rtol=0.0)
    @test isapprox(sort(μ̂), μ_true, atol=0.2, rtol=0.0)
end

# @testset "ising" begin
#     N = 20
#     β = 0.5
#     conditionals = isingdemo(β, N)
#     model = BlockModel(conditionals)
#     sampler = Gibbs(merge(NamedTuple(), [Symbol(:x, i) => 0 for i = 1:N]))
# end
