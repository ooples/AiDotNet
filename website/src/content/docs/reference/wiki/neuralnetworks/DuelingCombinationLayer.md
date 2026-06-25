---
title: "DuelingCombinationLayer<T>"
description: "Dueling DQN combination head (Wang et al."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Dueling DQN combination head (Wang et al. 2016, "Dueling Network
Architectures for Deep Reinforcement Learning"). Splits the shared
feature trunk's output into a scalar state-value V(s) and an
action-advantage vector A(s, a), then combines them as
Q(s, a) = V(s) + (A(s, a) − mean_a A(s, a))
to enforce the identifiability constraint Eq. 9 of the paper.

## For Beginners

"Dueling" means the network learns the value of
being in a state separately from the relative benefit of each action.
V(s) answers "is this state good?", A(s, a) answers "given this state,
is action a better or worse than average?". Combining them gives more
stable training than predicting Q(s, a) directly with one big head —
the value head doesn't have to redundantly encode "this state is good"
for every action.

## How It Works

**What this layer does:** Reads `[batch, featureDim]` shared
trunk features and emits `[batch, actionSize]` Q-values. Internally
holds two parallel linear projections — a single-unit value head and an
`actionSize`-unit advantage head — and combines them using the
mean-subtraction form (paper §3, Eq. 9). The mean form is what every
production Rainbow / Double-DQN implementation uses (e.g.
Stable-Baselines3 `DuelingQ`, RLlib `DuelingMixin`) because the
max-subtraction alternative is unstable under bootstrapping.

All forward arithmetic runs through `Engine`
ops on the trainable tensor instances, so the gradient tape captures
gradients for both heads automatically.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DuelingCombinationLayer(Int32,Int32,Nullable<Int32>)` | Initializes a new dueling combination head. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` |  |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` |  |
| `GetMetadata` | Constructor metadata for DeserializationHelper post-Clone reconstruction. |
| `GetParameters` |  |
| `GetTrainableParameters` |  |
| `ResetState` |  |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` |  |
| `UpdateParameters()` |  |

