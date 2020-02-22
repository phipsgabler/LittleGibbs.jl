# LittleGibbs.jl

A small implementation of Gibbs sampling for
[`AbstractMCMC.jl`](https://github.com/TuringLang/AbstractMCMC.jl) (“true” in the sense that it is
not a “within-Gibbs” sampler, but expected to be used with given true conditionals, which you have
to know or calculate).  Specifically, it is a “determinstic scan” implementation which in each
transition step cycles through all parameters, updating the each with its given conditional as in 

```
(θ_{1}, θ_{2}, ..., θ_{N}) <- (conditionals[1](θ_{2},...,θ_{N}), θ_{2}, ..., θ_{N})
(θ_{1}, θ_{2}, θ_{3:end}) <- (θ_{1}, conditionals[2](θ_{1}, θ_{3}, ..., θ_{N}), θ_{3}, ..., θ_{N})
...
(θ_{1}, ..., θ_{N-1}, θ_{N}) <- (θ_{1}, ..., θ_{N-1}, conditionals[N](θ_{1}, ..., θ_{N-1}))
```

(Of course you can actually use arbitrary componentwise proposal distributions, in reality.  But
that’s not the point.)


## Interface

As described in the [interface docs](https://turing.ml/dev/docs/using-turing/interface), there are
three types provided: 

- `Gibbs`, which defines the sampling algorithm
- `ConditionalModel`, which defines the probabilistic model in terms of its Gibbs conditionals
  (represented as a `NamedTuple` of functions returning distributions)
- `Transition`, which defines a step in the resulting chain (the parameters after a full update
  step, also as a `NamedTuple`)

As an example, for observations from a certain normal distribution with conjugate prior, we could
write the following:

```
# Generate data:
N = 100
x = π * randn(N)

# Prepare conditionals:
x̄ = mean(x)
cond_μ((σ²,)) = Normal(x̄, √(σ² / N))  # Normal is parametrized by σ, not σ²…
cond_σ²((μ,)) = InverseGamma(N / 2, N * var(x, mean=μ) / 2)

# write inference
model = BlockModel((μ=cond_μ, σ²=cond_σ²))
sampler = Gibbs((μ=0.0, σ²=1.0))
chain = sample(model, sampler, 10000);

mean(t -> t.params[:μ], chain)
mean(t -> t.params[:σ²], chain)
```

(This requires `Distributions` and `StatsBase`.)
