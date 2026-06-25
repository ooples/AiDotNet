---
title: "GLoRAAdapter"
description: "Generalized LoRA (GLoRA) implementation that adapts both weights AND activations."
section: "Reference"
---

_LoRA / PEFT Adapters_

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

