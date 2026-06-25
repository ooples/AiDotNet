---
title: "IntegratedGradientsExplainer<T>"
description: "Integrated Gradients explainer for neural networks with gradient access."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Integrated Gradients explainer for neural networks with gradient access.

## For Beginners

Integrated Gradients is a method for explaining neural network predictions
that satisfies two important mathematical properties:

1. **Completeness (Axiom of Completeness)**: The attributions sum up to the difference

between the prediction at the input and the prediction at a baseline (usually zeros).

2. **Sensitivity**: If a feature differs between input and baseline and affects the output,

it gets a non-zero attribution.

How it works:

- Start with a "baseline" (typically all zeros or a neutral input)
- Create a path from baseline to your actual input
- Integrate the gradients along this path
- The result shows how much each feature contributed to moving from baseline to final prediction

Why use Integrated Gradients?

- Theoretically sound (satisfies axioms that other methods don't)
- Works with any differentiable model
- Attributions have clear meaning: contribution to prediction difference from baseline

Example: For an image classifier predicting "cat":

- Baseline: black image (all zeros)
- Input: image of a cat
- Integrated Gradients shows which pixels contributed most to the "cat" prediction

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `IntegratedGradientsExplainer(Func<Vector<>,Vector<>>,Func<Vector<>,Int32,Vector<>>,Int32,Int32,Vector<>,String[])` | Initializes a new Integrated Gradients explainer. |
| `IntegratedGradientsExplainer(INeuralNetwork<>,Int32,Int32,Vector<>,String[])` | Initializes a new Integrated Gradients explainer from a neural network model. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsGPUAccelerated` |  |
| `MethodName` |  |
| `SupportsGlobalExplanations` |  |
| `SupportsLocalExplanations` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeGradient(Vector<>,Int32)` | Computes gradient either using provided function or numerical approximation. |
| `ComputeNumericalGradient(Vector<>,Int32)` | Computes numerical gradient using central difference. |
| `Explain(Vector<>)` | Computes Integrated Gradients attributions for an input. |
| `Explain(Vector<>,Int32)` | Computes Integrated Gradients attributions for a specific output class. |
| `ExplainBatch(Matrix<>)` |  |
| `SetGPUHelper(GPUExplainerHelper<>)` |  |

