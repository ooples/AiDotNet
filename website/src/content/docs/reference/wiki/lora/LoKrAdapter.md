---
title: "LoKrAdapter<T>"
description: "LoKr (Low-Rank Kronecker Product Adaptation) adapter for parameter-efficient fine-tuning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LoRA.Adapters`

LoKr (Low-Rank Kronecker Product Adaptation) adapter for parameter-efficient fine-tuning.

## For Beginners

LoKr is a variant of LoRA that uses a different mathematical operation
called the Kronecker product. Think of it this way:

- Standard LoRA: Multiplies two small matrices (like 1000×8 and 8×1000) to approximate changes
- LoKr: Uses Kronecker product of two even smaller matrices (like 50×4 and 20×4) to create the same size output

The Kronecker product creates a larger matrix by taking every element of the first matrix and
multiplying it by the entire second matrix. This creates a block pattern that's very efficient
for representing certain types of structured transformations.

**When to use LoKr vs standard LoRA:**

- LoKr is better for very wide or very deep layers (e.g., 10000×10000 weight matrices)
- LoKr can achieve similar expressiveness with fewer parameters than LoRA
- Standard LoRA is simpler and works well for typical layer sizes

**Parameter Efficiency Example:**
For a 1000×1000 weight matrix with rank r=8:

- Standard LoRA: 1000×8 + 8×1000 = 16,000 parameters
- LoKr: 50×4 + 20×4 = 200 + 80 = 280 parameters (57x fewer!)

(where 50×20 = 1000 for both dimensions)

## How It Works

LoKr uses Kronecker products instead of standard matrix multiplication for low-rank adaptation.
Instead of computing ΔW = A × B (standard LoRA), LoKr computes ΔW = A ⊗ B where ⊗ is the
Kronecker product. This is particularly efficient for very large weight matrices.

**Kronecker Product Definition:**
For matrices A (m×n) and B (p×q), the Kronecker product A ⊗ B is an (m×p) × (n×q) matrix:

A ⊗ B = [a₁₁B a₁₂B ... a₁ₙB]
[a₂₁B a₂₂B ... a₂ₙB]
[ ⋮ ⋮ ⋱ ⋮ ]
[aₘ₁B aₘ₂B ... aₘₙB]

Each element aᵢⱼ of A is multiplied by the entire matrix B, creating a block structure.

## Example

```csharp
using AiDotNet.LoRA;
using AiDotNet.LoRA.Adapters;

var adapter = new LoKrAdapter<double>(null, rank: 8, alpha: 8, freezeBaseLayer: true);
var config = new DefaultLoRAConfiguration<double>(rank: 8, alpha: 8, loraAdapter: adapter);
Console.WriteLine($"Configured LoKrAdapter (rank {config.Rank}).");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LoKrAdapter(ILayer<>,Int32,Double,Boolean)` | Initializes a new LoKr adapter wrapping an existing layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MatrixADimensions` | Gets the dimensions of matrix A. |
| `MatrixBDimensions` | Gets the dimensions of matrix B. |
| `ParameterCount` | Gets the total number of trainable parameters (elements in A and B matrices, plus base layer if not frozen). |

## Methods

| Method | Summary |
|:-----|:--------|
| `FactorDimension(Int32,Int32)` | Factors a dimension into two factors based on the desired rank. |
| `Forward(Tensor<>)` | Performs the forward pass through both base and LoKr layers. |
| `GetMatrixA` | Gets matrix A (for inspection or advanced use cases). |
| `GetMatrixB` | Gets matrix B (for inspection or advanced use cases). |
| `GetParameters` | Gets the current parameters as a vector. |
| `KroneckerGradientA(Matrix<>,Matrix<>,Matrix<>)` | Computes the gradient for matrix A using Kronecker product properties. |
| `KroneckerGradientB(Matrix<>,Matrix<>,Matrix<>)` | Computes the gradient for matrix B using Kronecker product properties. |
| `KroneckerProduct(Matrix<>,Matrix<>)` | Computes the Kronecker product of two matrices. |
| `MergeToOriginalLayer` | Merges the LoKr adaptation into the base layer and returns the merged layer. |
| `ResetState` | Resets the internal state of the adapter. |
| `SetParameters(Vector<>)` | Sets the layer parameters from a vector. |
| `UpdateMatricesFromParameters` | Updates the matrices from the parameter vector. |
| `UpdateMatricesWithGradients()` | Updates matrices A and B using their gradients. |
| `UpdateParameterGradientsFromMatrices` | Updates the parameter gradients vector from the matrix gradients. |
| `UpdateParameters()` | Updates the layer's parameters using the specified learning rate. |
| `UpdateParametersFromMatrices` | Updates the parameter vector from the current matrix states. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_alpha` | Scaling factor for the LoKr contribution. |
| `_dimsA` | Dimensions for matrix A (m, n). |
| `_dimsB` | Dimensions for matrix B (p, q). |
| `_gradientA` | Gradients for matrix A computed during backpropagation. |
| `_gradientB` | Gradients for matrix B computed during backpropagation. |
| `_lastInput` | Stored input from the forward pass, needed for gradient computation. |
| `_matrixA` | First Kronecker factor matrix A with dimensions (m × n). |
| `_matrixB` | Second Kronecker factor matrix B with dimensions (p × q). |
| `_scaling` | Computed scaling factor (alpha / effective_rank) used during forward pass. |

