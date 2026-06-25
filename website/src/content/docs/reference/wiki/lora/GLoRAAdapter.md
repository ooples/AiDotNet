---
title: "GLoRAAdapter<T>"
description: "Generalized LoRA (GLoRA) implementation that adapts both weights AND activations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LoRA.Adapters`

Generalized LoRA (GLoRA) implementation that adapts both weights AND activations.

## For Beginners

While standard LoRA only adapts what the layer learns (its weights),
GLoRA also adapts what the layer produces (its activations). Think of it like this:

- Standard LoRA: Adjusts the "recipe" (weights) but produces the same type of output
- GLoRA: Adjusts both the "recipe" (weights) AND transforms the output for different uses

This is especially useful when:

1. Different tasks need different feature representations
2. You're doing multi-task learning (e.g., the same base features used differently)
3. You need more flexibility than weight-only adaptation provides

Key differences from StandardLoRA:

- WeightAdaptation: Standard LoRA component that modifies layer weights
- ActivationAdaptation: Additional LoRA component that modifies layer outputs
- ActivationRank: Can be different from weight rank for fine-tuned control

Trade-offs:
+ More flexible: Can adapt representations for different tasks
+ Better for multi-task: Each task can use features differently

- More parameters: Two LoRA components instead of one
- Slightly slower: Two adaptation computations per forward pass

Example: For a 1000x1000 layer with weight_rank=8 and activation_rank=4:

- Weight adaptation: 16,000 parameters (same as standard LoRA)
- Activation adaptation: 8,000 additional parameters
- Total: 24,000 parameters (still 97.6% reduction from 1M!)

## How It Works

GLoRA extends standard LoRA by adding adaptation to both the layer's weights and its activations.
This provides more flexibility for multi-task learning scenarios where different tasks may need
different feature representations at each layer.

The forward pass computes:

- adapted_weights = base_weights + B_w * A_w (weight adaptation)
- base_output = input * adapted_weights
- adapted_output = base_output + B_a * A_a * input (activation adaptation)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GLoRAAdapter(ILayer<>,Int32,Int32,Double,Double,Boolean)` | Initializes a new GLoRA adapter with the specified parameters. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ActivationAdaptation` | Gets the activation adaptation LoRA layer. |
| `ActivationRank` | Gets the rank of the activation adaptation. |
| `ParameterCount` | Gets the total number of trainable parameters (both weight and activation adaptations). |
| `WeightAdaptation` | Gets the weight adaptation LoRA layer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Performs the forward pass through both base layer and both LoRA adaptations. |
| `GetParameters` | Gets the current parameters as a vector. |
| `MergeToOriginalLayer` | Merges both LoRA adaptations into the base layer and returns the merged layer. |
| `ResetState` | Resets the internal state of the base layer and both LoRA adaptations. |
| `SetParameters(Vector<>)` | Sets the layer parameters from a vector. |
| `UpdateLayersFromParameters` | Updates the layers from the parameter vector. |
| `UpdateParameterGradientsFromLayers` | Updates the parameter gradients vector from the layer gradients. |
| `UpdateParameters()` | Updates parameters using the specified learning rate. |
| `UpdateParametersFromLayers` | Updates the parameter vector from the current layer states. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_activationAdaptation` | The LoRA layer that adapts activations (layer outputs). |

