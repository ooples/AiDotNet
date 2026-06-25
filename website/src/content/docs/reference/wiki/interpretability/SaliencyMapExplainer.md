---
title: "SaliencyMapExplainer<T>"
description: "Saliency Map explainer using gradient-based methods."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Saliency Map explainer using gradient-based methods.

## For Beginners

Saliency maps are one of the simplest ways to explain neural networks.
They show which input features are most "sensitive" - where small changes would most affect the output.

Types of saliency methods:

1. **Vanilla Gradient**: The raw gradient of output w.r.t. input
2. **Gradient × Input**: Gradient multiplied by input (more interpretable)
3. **SmoothGrad**: Average gradient over noisy versions (reduces noise)
4. **SmoothGrad²**: Squared gradients for sharper feature focus

How to interpret:

- High absolute saliency = changing this feature would change the output a lot
- For images: bright spots show important pixels
- For tabular data: high values show important features

Pros:

- Fast to compute (single backward pass)
- Easy to understand
- Works with any differentiable model

Cons:

- Can be noisy (especially vanilla gradient)
- Doesn't show actual contribution, just sensitivity
- Can miss important features with low gradient

SmoothGrad is recommended for cleaner visualizations.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SaliencyMapExplainer(Func<Vector<>,Vector<>>,Func<Vector<>,Int32,Vector<>>,Int32,SaliencyMethod,Int32,Double,String[],Nullable<Int32>)` | Initializes a new Saliency Map explainer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MethodName` |  |
| `SupportsGlobalExplanations` |  |
| `SupportsLocalExplanations` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeGradient(Vector<>,Int32)` | Computes gradient using provided function or numerical approximation. |
| `ComputeGradientTimesInput(Vector<>,Int32)` | Computes gradient × input. |
| `ComputeNumericalGradient(Vector<>,Int32)` | Computes numerical gradient. |
| `ComputeSmoothGrad(Vector<>,Int32)` | Computes SmoothGrad (average gradient over noisy inputs, then multiplied by input). |
| `ComputeSmoothGradSquared(Vector<>,Int32)` | Computes SmoothGrad² (squared gradients for sharper focus). |
| `ComputeVanillaGradient(Vector<>,Int32)` | Computes vanilla gradient. |
| `Explain(Vector<>)` | Computes saliency map for an input. |
| `Explain(Vector<>,Int32)` | Computes saliency map for a specific output. |
| `ExplainBatch(Matrix<>)` |  |
| `GetArgMax(Vector<>)` | Gets the index of the maximum value. |

