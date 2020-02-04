module LittleGibbs

using AbstractMCMC
import AbstractMCMC: step!, transition_type
using Random

export Gibbs, BlockModel, Transition



struct Gibbs{B, T<:NamedTuple} <: AbstractSampler
    init_θ::T
    
    Gibbs(init_θ::NamedTuple{B, D}) where {B, D} = new{B, NamedTuple{B, D}}(init_θ)
end

struct BlockModel{B, C<:NamedTuple} <: AbstractModel
    conditionals::C
    
    BlockModel(conditionals::NamedTuple{B, D}) where {B, D} = new{B, NamedTuple{B, D}}(conditionals)
end

struct Transition{B, T<:NamedTuple} <: AbstractTransition
    params::T

    Transition(params::NamedTuple{B, D}) where {B, D} = new{B, NamedTuple{B, D}}(params)
end


transition_type(model::BlockModel{B, T}, spl::Gibbs{B}) where {B, T} = Transition{B, T}


function step!(rng::AbstractRNG, model::BlockModel{B}, spl::Gibbs{B},
               ::Integer; kwargs...) where {B}
    return Transition(spl.init_θ)
end

function step!(rng::AbstractRNG, model::BlockModel{B}, spl::Gibbs{B, T},
               ::Integer, θ_prev::Transition{B, T}; kwargs...) where {B, T}
    params = θ_prev.params
    conditionals = model.conditionals
    return Transition(updated(rng, conditionals, params))
    # return propose(rng, spl, model, θ_prev)
end


updated(rng, conditionals::NamedTuple{()}, params::NamedTuple) = params

function updated(rng, conditionals::NamedTuple{B}, params::NamedTuple) where B
    conditional_dist = first(conditionals)
    conditioned_values = Base.tail(params)
    return updated(rng, Base.tail(conditionals),
                   merge(conditioned_values,
                        (; first(B) => rand(rng, conditional_dist(conditioned_values)))))
end


end # module
