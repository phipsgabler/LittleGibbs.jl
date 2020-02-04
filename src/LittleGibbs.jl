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
    return propose(rng, spl, model, θ_prev)
end


function propose(rng::AbstractRNG, spl::Gibbs{B, T}, model::BlockModel{B},
                 θ_prev::Transition{B, T}) where {B, T}
    params = θ_prev.params

    for block in B
        conditional_dist = model.conditionals[block]
        params = conditionally(params, Val(block)) do conditioned
            rand(rng, conditional_dist(conditioned))
        end
    end
    
    return Transition(params)
end

@generated function conditionally(f, t::NamedTuple{B, T}, ::Val{b}) where {B, T, b}
    conditioned_values = Expr[]
    updated_values = Expr[]
    @gensym update
    
    for block in B
        if block == b
            push!(updated_values, :($block = $update))
        else
            part = :($block = t[$(QuoteNode(block))])
            push!(updated_values, part)
            push!(conditioned_values, part)
        end
    end
    
    return quote
        let $update = f((;$(conditioned_values...)))
            (;$(updated_values...))
        end
    end
end


end # module
