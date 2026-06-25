---
title: "DoRAAdapter<T>"
description: "DoRA (Weight-Decomposed Low-Rank Adaptation) adapter for parameter-efficient fine-tuning with improved stability."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LoRA.Adapters`

DoRA (Weight-Decomposed Low-Rank Adaptation) adapter for parameter-efficient fine-tuning with improved stability.

## For Beginners

DoRA is an improved version of LoRA that works better in practice.

Think of neural network weights as arrows:

- Each arrow has a length (magnitude) and a direction
- Standard LoRA adjusts both length and direction at the same time
- DoRA separates them: it keeps the length fixed and only adjusts the direction
- This makes training more stable and gives better results

Why this matters:

- More stable training (fewer divergences and NaN errors)
- Better final performance (+3.7% on LLaMA-7B)
- Same parameter efficiency as standard LoRA
- Slightly more computation (due to normalization), but worth it for the stability

When to use DoRA over standard LoRA:

- When training stability is important (large models, complex tasks)
- When you want the best possible fine-tuning results
- When you have the computational budget for normalization overhead
- When adapting very large pre-trained models (LLMs, large vision models)

## How It Works

DoRA (Weight-Decomposed LoRA) extends standard LoRA by decomposing pre-trained weights into
magnitude and direction components, then applying LoRA only to the direction component.
This decomposition leads to more stable training and better convergence compared to standard LoRA.

**Mathematical Formulation:**
Given pre-trained weights W, DoRA decomposes them as:

- W = m * d, where m is magnitude (scalar per neuron) and d is direction (unit vector)
- W' = m * normalize(d + LoRA_delta)
- LoRA_delta = (alpha/rank) * B * A

This ensures that LoRA adaptations primarily affect the direction of weights, not their magnitude,
which improves training stability and convergence.

**Research Context:**
DoRA was published in February 2024 and presented as an ICML 2024 Oral paper.
In experiments on LLaMA-7B, DoRA achieved +3.7% improvement over standard LoRA.
The key insight is that separating magnitude and direction allows more stable gradient flow
and better control over the adaptation process.

**Reference:**
"DoRA: Weight-Decomposed Low-Rank Adaptation"
ICML 2024 Oral
https://arxiv.org/abs/2402.09353

## Example

```csharp
using AiDotNet.LoRA;
using AiDotNet.LoRA.Adapters;

var adapter = new DoRAAdapter<double>(null, rank: 8, alpha: 8, freezeBaseLayer: true);
var config = new DefaultLoRAConfiguration<double>(rank: 8, alpha: 8, loraAdapter: adapter);
Console.WriteLine($"Configured DoRAAdapter (rank {config.Rank}).");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DoRAAdapter(ILayer<>,Int32,Double,Boolean)` | Initializes a new DoRA adapter wrapping an existing layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets the total number of trainable parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DecomposeWeights` | Decomposes the base layer's weights into magnitude and direction components. |
| `Forward(Tensor<>)` | Performs the forward pass through DoRA adapter. |
| `GetParameters` | Gets the current parameters as a vector. |
| `MergeToOriginalLayer` | Merges the DoRA adaptation into the base layer and returns the merged layer. |
| `NormalizeRows(Matrix<>)` | Normalizes a matrix row-wise (each row becomes a unit vector). |
| `RecomposeWeights(Matrix<>)` | Recomposes weights from magnitude and direction components. |
| `ResetState` | Resets the internal state of the adapter. |
| `SetParameters(Vector<>)` | Sets the layer parameters from a vector. |
| `UpdateComponentsFromParameters` | Updates the components from the parameter vector. |
| `UpdateParameters()` | Updates parameters using the specified learning rate. |
| `UpdateParametersFromComponents` | Updates the parameter vector from the current component states. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_lastInputMatrix` | Cached input matrix from forward pass (used for computing magnitude gradients in backward). |
| `_lastNormalizedDirection` | Cached normalized direction from the last forward pass, used in backpropagation. |
| `_magnitude` | Magnitude component of the decomposed weights (scalar per output neuron). |
| `_magnitudeGradient` | Gradients for the magnitude component, computed during backpropagation. |

