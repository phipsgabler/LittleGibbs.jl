module LittleGibbs

import MCMCChains: Chains
import AbstractMCMC: step!, AbstractSampler, AbstractTransition, transition_type, bundle_samples
using Distributions
using Random


struct Gibbs{B, T<:NamedTuple} <: AbstractSampler
    init_θ::T
    
    Gibbs(init_θ::NamedTuple{B}) where {B} = new{B, NamedTuple{B}}(init_θ)
end

struct BlockModel{B, C<:NamedTuple} <: AbstractModel
    conditionals::C
    
    BlockModel(conditionals::NamedTuple{B}) where {B} = new{B, NamedTuple{B}}(conditionals)
end

struct Transition{B, T<:NamedTuple} <: AbstractTransition
    params::T

    Transition(params::NamedTuple{B}, block::Symbol) where {B} = new{B, NamedTuple{B}}(params)
end


transition_type(model::BlockModel{B, T}, spl::Gibbs{B}) where {B, T} = Transition{B, T}

function step!(rng::AbstractRNG, model::BlockModel{B}, spl::Gibbs{B},
               ::Integer; kwargs...) where {B}
    return Transition(model.init_θ)
end

function step!(rng::AbstractRNG, model::BlockModel{B}, spl::Gibbs{B, T},
               ::Integer, θ_prev::Transition{B, T}; kwargs...) where {B, T}
    return propose(rng, spl, model, θ_prev)
end

function propose(rng::AbstractRNG, spl::Gibbs{B, T}, model::BlockModel{B},
                 θ_prev::Transition{B, T}) where {B, T}
    params = deepcopy(θ_prev.params)

    for block in B
        conditional = model.conditionals[block]
        rand!(rng, params[b], conditional(without(params, Val(block))))
    end
    
    return Transition(params)
end

@generated function without(t::NamedTuple{B, T}, ::Val{n}) where {B, T, n}
    parts = (:($m = t[$(QuoteNode(m))]) for m in B if m ≠ n)
    return :(;$(parts...))
end


function bundle_samples(rng::AbstractRNG, ::BlockModel{B, T}, ::Gibbs{B}, ::Integer,
                        ts::Vector{Transition{B, T}}; kwargs...) where {B, T}
    vals = copy(reduce(hcat, [values(t.params) for t in ts])')
    return Chains(vals, B)
end


end # module
