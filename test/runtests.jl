using Test
using HypothesisTests
using LittleGibbs

include("./testmodels.jl")


@testset "gdemo" begin
    λ_true, m_true = 1/3, 15
    α₀, β₀ = 2.0, 3.0
    N = 30
    # conditionals, posterior_marginals = gdemo(sqrt(1/λ_true) .* randn(N) .+ m_true; α₀ = α₀, β₀ = β₀)
    conditionals, posterior_marginals = gdemo([1.5, 2.0])
    model = BlockModel(conditionals)
    sampler = Gibbs((m = 0.0, λ = 1.0))
    
    chain = sample(model, sampler, 11_000)
    chain_m = map(t -> t.params[:m], chain[1000:end])
    chain_λ = map(t -> t.params[:λ], chain[1000:end])
    
    m̂ = mean(chain_m)
    λ̂ = mean(chain_λ)
    @info (m̂ = m̂, λ̂ = λ̂)
    
    # x = y = -2:0.05:2
    # @show exact_posterior(-1.5, -1.5)
    # display(contour(x, y, exact_posterior))
    
    p_m = pvalue(ExactOneSampleKSTest(chain_m, posterior_marginals[:m]))
    p_λ = pvalue(ExactOneSampleKSTest(chain_λ, posterior_marginals[:λ]))
    @test p_m < 0.05
    @test p_λ < 0.05
    @info (p_m = p_m, p_λ = p_λ)
    # dump(ExactOneSampleKSTest(chain_m, posterior_marginals[:m]))
end

@testset "ising" begin
    N = 20
    conditionals = ising(N, 0.5)
    model = BlockModel(conditionals)
    sampler = Gibbs(merge(NamedTuple(), [Symbol(:x, i) => 0 for i = 1:N]))
end
