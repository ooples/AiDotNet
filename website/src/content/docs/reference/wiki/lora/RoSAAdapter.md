---
title: "RoSAAdapter<T>"
description: "RoSA (Robust Adaptation) adapter for parameter-efficient fine-tuning with improved robustness to distribution shifts."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LoRA.Adapters`

RoSA (Robust Adaptation) adapter for parameter-efficient fine-tuning with improved robustness to distribution shifts.

## For Beginners

RoSA is like LoRA with a safety net for unusual cases.

Think of it this way:

- Low-rank LoRA is like learning general rules ("most images of cats have pointed ears")
- Sparse component is like remembering specific exceptions ("this one cat breed has round ears")
- Together they make a robust model that handles both common and rare cases

Why RoSA is more robust:

- Low-rank component: Efficient for common patterns across domains
- Sparse component: Handles outliers and domain-specific quirks
- Result: Better performance when test data differs from training data

When to use RoSA over standard LoRA:

- When you expect distribution shifts (train on news, test on social media)
- When your data has outliers or rare patterns that matter
- When you need robustness more than absolute parameter efficiency
- When adapting to multiple related but distinct domains

Trade-offs vs standard LoRA:
+ More robust to distribution shifts
+ Better handles rare patterns
+ More flexible adaptation

- Slightly more parameters (sparse component adds ~5-15%)
- Slightly more computation (extra sparse matrix multiply)
- Requires tuning sparsity ratio

## How It Works

RoSA (Robust Adaptation) extends standard LoRA by combining two complementary components:

1. Low-rank component (standard LoRA): Captures common, structured patterns in adaptations
2. Sparse component: Captures specific, rare, or outlier patterns that low-rank cannot represent

**Mathematical Formulation:**
Given input x and pre-trained weights W, RoSA computes:

- Low-rank component: L = (alpha/rank) * B * A * x
- Sparse component: S = W_sparse * x (where W_sparse is highly sparse)
- Final output: y = W*x + L + S

The sparse component is maintained through magnitude-based pruning, keeping only the
most significant weights and zeroing out the rest. This creates a sparse matrix that
captures specific patterns while remaining parameter-efficient.

**Research Context:**
RoSA was introduced in January 2024 as a robust alternative to standard LoRA.
The key insight is that low-rank approximations work well for common patterns but
struggle with distribution shifts and rare patterns. By adding a sparse component,
RoSA can capture outliers and domain-specific patterns without significantly
increasing parameter count.

In experiments on domain adaptation tasks, RoSA showed:

- Better generalization to new domains (+5-10% over standard LoRA)
- More robust to distribution shifts
- Ability to capture both global patterns (low-rank) and local exceptions (sparse)
- Only modest increase in parameters (typically 5-15% more than pure LoRA)

**Reference:**
"RoSA: Robust Adaptation through Sparse Regularization"
January 2024

## Example

```csharp
using AiDotNet.LoRA;
using AiDotNet.LoRA.Adapters;

var adapter = new RoSAAdapter<double>(null, rank: 8, alpha: 8, freezeBaseLayer: true);
var config = new DefaultLoRAConfiguration<double>(rank: 8, alpha: 8, loraAdapter: adapter);
Console.WriteLine($"Configured RoSAAdapter (rank {config.Rank}).");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RoSAAdapter(ILayer<>,Int32,Double,Double,Double,Boolean)` | Initializes a new RoSA adapter wrapping an existing layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets the total number of trainable parameters. |
| `SparseThreshold` | Threshold for magnitude-based pruning of sparse weights. |
| `SparsityRatio` | Target sparsity ratio (fraction of zeros in sparse component). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Performs the forward pass through RoSA adapter. |
| `GetParameters` | Gets the current parameters as a vector. |
| `GetSparsity` | Gets the current sparsity of the sparse component. |
| `InitializeSparseWeights` | Initializes sparse weights with small random values. |
| `MergeToOriginalLayer` | Merges the RoSA adaptation into the base layer and returns the merged layer. |
| `PruneSparseWeights` | Prunes sparse weights based on magnitude to maintain target sparsity. |
| `ResetState` | Resets the internal state of the adapter. |
| `SetParameters(Vector<>)` | Sets the layer parameters from a vector. |
| `UpdateComponentsFromParameters` | Updates the components from the parameter vector. |
| `UpdateParameters()` | Updates parameters using the specified learning rate. |
| `UpdateParametersFromComponents` | Updates the parameter vector from the current component states. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_cachedInputMatrix` | Cached input matrix from forward pass, needed for computing sparse weight gradients. |
| `_sparseGradients` | Gradients for the sparse weight component, computed during backpropagation. |
| `_sparseWeights` | Sparse weight matrix that captures specific/rare patterns. |

