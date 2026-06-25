---
title: "DeltaLoRAAdapter<T>"
description: "Delta-LoRA adapter that focuses on parameter-efficient delta updates with momentum."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LoRA.Adapters`

Delta-LoRA adapter that focuses on parameter-efficient delta updates with momentum.

## For Beginners

Think of Delta-LoRA as "change-focused" LoRA.

Regular LoRA learns: "What should the weights be?"
Delta-LoRA learns: "How should the weights change?"

This difference matters because:

1. Changes (deltas) often have simpler patterns than absolute values
2. Momentum helps smooth out noisy updates
3. Can converge faster when the optimal adaptation is a smooth transformation

Key concepts:

- **Delta weights**: Accumulated changes to parameters (not the parameters themselves)
- **Delta scaling**: Controls how strongly deltas affect the output
- **Momentum**: Smooths updates by remembering previous changes

When Delta-LoRA works better than standard LoRA:

- Tasks requiring smooth, gradual adaptations
- Fine-tuning where the base model is already close to optimal
- Scenarios with noisy gradients that benefit from momentum
- Transfer learning where you want to preserve more of the original model's behavior

Example: If you're adapting a language model to a new domain, Delta-LoRA can
make smaller, more conservative changes that preserve the model's general knowledge
while adapting to domain-specific patterns.

## How It Works

Delta-LoRA is a variant of LoRA that explicitly models the change (delta) in parameters
rather than the absolute values. This approach can achieve better convergence in certain
scenarios by focusing on the parameter update dynamics with momentum-based accumulation.

## Example

```csharp
using AiDotNet.LoRA;
using AiDotNet.LoRA.Adapters;

var adapter = new DeltaLoRAAdapter<double>(null, rank: 8, alpha: 8, freezeBaseLayer: true);
var config = new DefaultLoRAConfiguration<double>(rank: 8, alpha: 8, loraAdapter: adapter);
Console.WriteLine($"Configured DeltaLoRAAdapter (rank {config.Rank}).");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeltaLoRAAdapter(ILayer<>,Int32,Double,Double,Double,Boolean)` | Initializes a new Delta-LoRA adapter wrapping an existing layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DeltaScaling` | Gets the scaling factor for delta updates. |
| `MomentumFactor` | Gets the momentum factor for delta accumulation. |
| `ParameterCount` | Gets the total number of trainable parameters including delta weights. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Performs the forward pass: output = base_layer(input) + LoRA(input) + delta_weights @ input * delta_scaling. |
| `GetCurrentDelta` | Gets the current delta weights matrix. |
| `GetParameterGradients` | Gets all parameter gradients including base layer, LoRA layer, and delta weight gradients. |
| `GetParameters` | Gets the current parameters including base layer, LoRA layer, and delta weights. |
| `MergeToOriginalLayer` | Merges the LoRA adaptation and delta weights into the base layer. |
| `ResetState` | Resets the internal state including delta weights, velocity, and cached inputs. |
| `SetParameters(Vector<>)` | Sets the layer parameters including base layer, LoRA layer, and delta weights. |
| `UpdateParameters()` | Updates parameters using momentum-based delta updates. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_deltaGradients` | Gradients for the delta weights computed during backpropagation. |
| `_deltaScaling` | Scaling factor applied to delta updates before adding to the output. |
| `_deltaWeights` | Matrix storing the cumulative weight deltas (changes over time). |
| `_lastInput` | Stored input from the forward pass, needed for gradient computation. |
| `_momentumFactor` | Momentum factor for delta accumulation (0 to 1). |
| `_velocity` | Velocity matrix for momentum-based updates. |

