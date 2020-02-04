module LittleGibbs

import MCMCChains: Chains
import AbstractMCMC: step!, AbstractSampler, AbstractTransition, transition_type, bundle_samples
using Distributions
using Random

struct Gibbs{N, T<:NamedTuple} <: AbstractSampler
    init_θ::T
    
    Gibbs(init_θ::NamedTuple{N}) where {N} = new{N, NamedTuple{N}}(init_θ)
end

struct BlockModel{N, C<:NamedTuple} <: AbstractModel
    conditionals::C
    
    BlockModel(conditionals::NamedTuple{N}) where {N} = new{N, NamedTuple{N}}(conditionals)
end

struct Transition{N, T<:NamedTuple} <: AbstractTransition
    params::T

    Transition(params::NamedTuple{N}) where {N} = new{N, NamedTuple{N}}(params)
end


transition_type(model::BlockModel{N, T}, spl::Gibbs{N}) where {N} = Transition{T}

function step!(rng::AbstractRNG, model::BlockModel{N}, spl::Gibbs{N},
               ::Integer; kwargs...) where {N}
    return Transition(model.init_θ)
end

function step!(rng::AbstractRNG, model::BlockModel{N}, spl::Gibbs{N},
               n::Integer, θ_prev::Transition{N}; kwargs...) where {N}
    return propose(rng, spl, model, θ_prev, n)
end

function propose(rng::AbstractRNG, spl::Gibbs{N}, model::BlockModel{N},
                 θ_prev::Transition{N}, n::Int)
    block = N[mod1(n, length(N))]
    conditional = model.conditionals[block]
    params = rand(rng, conditional(without(θ_prev, Val(block))))
    return Transition(params)
end

@generated function without(t::NamedTuple{N, T}, ::Val{n}) where {N, T, n}
    parts = (:($m = t[$(QuoteNode(m))]) for m in N if m ≠ n)
    return :(;$(parts...))
end


end # module
