---
title: "CASTLEAlgorithm<T>"
description: "CASTLE — Causal Structure Learning via neural networks with shared masked architecture."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.DeepLearning`

CASTLE — Causal Structure Learning via neural networks with shared masked architecture.

## For Beginners

CASTLE trains a little predictor for each variable that guesses
it from the others. How strongly each other variable feeds into that predictor tells
us how likely it is a cause. A sparsity penalty drops weak links and an acyclicity
rule keeps the result a valid causal graph (no loops).

## How It Works

CASTLE trains one neural sub-network f_j per variable to reconstruct x_j from the
other variables. The causal adjacency is read directly from each sub-network's
first-layer weights — A[i,j] = ‖Wh_j[i,:]‖, the influence of variable i on the
reconstruction of variable j — rather than from a separate gate. Sparsity is imposed
by a group-lasso penalty on those input-weight rows, and acyclicity by the NOTEARS
constraint h(A) = tr(exp(A∘A)) − d = 0, enforced with an augmented Lagrangian.

**Algorithm:**

- Standardize the data; initialize one MLP f_j per target (self-input held at 0)
- Forward: H = σ(X·Wh_j), pred = H·Wo_j; reconstruction MSE against x_j
- After a reconstruction-first warmup, add group-L1 on the Wh_j rows and the

augmented-Lagrangian acyclicity gradient on A_sq[i,j] = ‖Wh_j[i,:]‖²

- Update all weights with Adam; dual-ascend α and escalate ρ to drive h→0
- Read adjacency from the final weight-row norms, normalize, and threshold

Reference: Kyono et al. (2020), "CASTLE: Regularization via Auxiliary Causal Graph Discovery", NeurIPS.

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |

