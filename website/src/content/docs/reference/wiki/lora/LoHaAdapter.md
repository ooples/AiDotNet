---
title: "LoHaAdapter<T>"
description: "LoHa (Low-Rank Hadamard Product Adaptation) adapter for parameter-efficient fine-tuning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LoRA.Adapters`

LoHa (Low-Rank Hadamard Product Adaptation) adapter for parameter-efficient fine-tuning.

## For Beginners

LoHa is a variant of LoRA that uses element-wise multiplication
instead of matrix multiplication. Think of it this way:

- Standard LoRA: Learns "row and column patterns" that combine via matrix multiply
- LoHa: Learns "pixel-by-pixel patterns" that combine via element-wise multiply

LoHa is especially good when:

1. You need to capture local, element-wise patterns (like in images)
2. The weight matrix has spatial structure (like convolutional filters)
3. You want each weight to be adjusted somewhat independently

Trade-offs compared to LoRA:

- More parameters: Both A and B must be full-sized (input×output) per rank dimension
- Different expressiveness: Better for element-wise patterns, different from matrix patterns
- Better for CNNs: The element-wise nature matches convolutional structure better

Example: A 100×100 weight matrix with rank=8

- Standard LoRA: 8×100 + 100×8 = 1,600 parameters
- LoHa: 2 × 8 × 100 × 100 = 160,000 parameters (each rank has 2 full-sized matrices)

LoHa uses MORE parameters than LoRA but models element-wise weight interactions via Hadamard products.

## How It Works

LoHa uses element-wise Hadamard products (⊙) instead of matrix multiplication for adaptation.
Instead of computing ΔW = B * A like standard LoRA, LoHa computes:
ΔW = sum over rank of (A[i] ⊙ B[i])

This formulation can capture element-wise patterns that matrix multiplication may miss,
making it particularly effective for:

- Convolutional layers (local spatial patterns)
- Element-wise transformations
- Fine-grained weight adjustments

**Mathematical Formulation:**

Standard LoRA: ΔW = B * A where B is rank×output, A is input×rank
LoHa: ΔW = Σ(A[i] ⊙ B[i]) where A[i] and B[i] are both input×output

The Hadamard product (⊙) performs element-wise multiplication, allowing each element
of the weight matrix to be adjusted independently across the rank dimensions.

## Example

```csharp
using AiDotNet.LoRA;
using AiDotNet.LoRA.Adapters;

var adapter = new LoHaAdapter<double>(null, rank: 8, alpha: 8, freezeBaseLayer: true);
var config = new DefaultLoRAConfiguration<double>(rank: 8, alpha: 8, loraAdapter: adapter);
Console.WriteLine($"Configured LoHaAdapter (rank {config.Rank}).");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LoHaAdapter(ILayer<>,Int32,Double,Boolean)` | Initializes a new LoHa adapter wrapping an existing layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets the total number of trainable parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLoHaDelta(Tensor<>)` | Computes the LoHa delta using Hadamard products across all rank dimensions. |
| `ComputeLoHaGradients(Tensor<>)` | Computes gradients for LoHa matrices A and B using Hadamard product gradient rules. |
| `Forward(Tensor<>)` | Performs the forward pass through both base layer and LoHa adaptation. |
| `GetParameters` | Gets the current parameters as a vector. |
| `MergeToOriginalLayer` | Merges the LoHa adaptation into the base layer and returns the merged layer. |
| `ResetState` | Resets the internal state of both the base layer and LoHa adapter. |
| `SetParameters(Vector<>)` | Sets the layer parameters from a vector. |
| `UpdateMatricesFromParameters` | Updates the matrices from the parameter vector. |
| `UpdateParameterGradientsFromMatrices` | Updates the parameter gradients vector from the matrix gradients. |
| `UpdateParameters()` | Updates parameters using the specified learning rate. |
| `UpdateParametersFromMatrices` | Updates the parameter vector from the current matrix values. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_lastBaseOutput` | Stored base layer output from the forward pass. |
| `_lastInput` | Stored input from the forward pass, needed for gradient computation. |
| `_matricesA` | Low-rank matrices A with dimensions (rank, inputSize, outputSize). |
| `_matricesAGradient` | Gradients for matrices A computed during backpropagation. |
| `_matricesB` | Low-rank matrices B with dimensions (rank, inputSize, outputSize). |
| `_matricesBGradient` | Gradients for matrices B computed during backpropagation. |
| `_scaling` | Computed scaling factor (alpha / rank) used during forward pass. |

