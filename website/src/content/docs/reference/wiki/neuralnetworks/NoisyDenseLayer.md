---
title: "NoisyDenseLayer<T>"
description: "Noisy linear layer for exploration in reinforcement learning (Fortunato et al."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Noisy linear layer for exploration in reinforcement learning (Fortunato et al.
2017 "Noisy Networks for Exploration", §3.2 Factorised Gaussian variant).
Replaces a conventional dense layer's deterministic weights with parametric
noise: `W = μ_w + σ_w ⊙ ε_w`, `b = μ_b + σ_b ⊙ ε_b`, where ε is
resampled on every forward pass. σ is learned jointly with μ, so the network
decides per-weight how much exploration noise to inject.

## For Beginners

A regular dense (fully-connected) layer
learns one weight per (input, output) pair. NoisyDenseLayer learns TWO
per pair — a base value `μ` and a noise scale `σ` — and at
training time draws a fresh random ε every forward pass to form the
effective weight `W = μ + σ · ε`. The network learns when to make
σ small (confident, deterministic predictions) vs large (uncertain,
exploring different actions). This replaces hand-tuned exploration
strategies like ε-greedy in reinforcement-learning agents — the
exploration noise is built into the weights and decays naturally as
training converges. At evaluation time σ is zeroed out (paper §3.4)
so the network is deterministic given a fixed state.

## How It Works

Factorised noise generates two independent Gaussian vectors ε_in (size p)
and ε_out (size q), applies `f(x) = sign(x)·√|x|` to each, and forms
the weight noise matrix as the outer product
`ε_w[i,j] = f(ε_in[i]) · f(ε_out[j])`. This needs p+q random draws
instead of p·q for the independent variant (paper §3.2).

Initialisation follows Fortunato 2017 Eqs. 17–18:
`μ ~ U(-1/√p, 1/√p)`, `σ_init = 0.5/√p` for both weights and biases.

All forward arithmetic is routed through `Engine`
ops on the same tensor instances returned by `GetTrainableParameters`,
so the gradient tape automatically captures gradients with respect to μ_w,
σ_w, μ_b, and σ_b. The ε tensors are rebuilt per forward and are NOT
trainable — the tape treats them as input data.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NoisyDenseLayer(Int32,Int32,IActivationFunction<>,Nullable<Double>,Nullable<Int32>)` | Creates a new noisy dense layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` |  |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` |  |
| `GetMetadata` | Persists the constructor parameters needed by `DeserializationHelper` to reconstruct an identical layer post-Clone. |
| `GetParameters` |  |
| `GetTrainableParameters` |  |
| `ResetState` |  |
| `SampleFactorisedNoise` | Resamples ε_in (size p) and ε_out (size q), applies the signed-sqrt transform `f(x) = sign(x)·√\|x\|`, and builds the per-forward noise tensors via an engine-accelerated outer product: `ε_w = f(ε_in)ᵀ · f(ε_out)` (Fortunato 2017 §3.2). |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` |  |
| `SignedSqrt(Tensor<>)` | Element-wise signed-square-root: `f(x) = sign(x) · √\|x\|`, the per-element transform Fortunato 2017 §3.2 specifies for factorised noise. |
| `UpdateParameters()` |  |

