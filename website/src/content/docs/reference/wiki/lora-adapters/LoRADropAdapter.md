---
title: "LoRADropAdapter"
description: "LoRA-drop implementation: LoRA with dropout regularization."
section: "Reference"
---

_LoRA / PEFT Adapters_

LoRA-drop implementation: LoRA with dropout regularization.

## For Beginners

LoRA-drop adds dropout regularization to LoRA adapters.

Dropout is a technique where during training, we randomly "turn off" some neurons or components.
This prevents the model from becoming too dependent on specific components and forces it to
learn more general patterns.

Think of it like practicing a skill with random handicaps:

- Sometimes you practice with your left hand tied behind your back
- Sometimes you practice blindfolded
- This forces you to develop multiple strategies instead of relying on one approach

LoRA-drop applies this to LoRA adaptations:

- During training: Randomly drop some LoRA components (set them to zero)
- During inference: Use all components but scale them appropriately
- Result: More robust adaptations that generalize better to new data

Recommended dropout rates:

- 0.1 (10%): Light regularization, good starting point
- 0.2 (20%): Moderate regularization, common choice
- 0.3 (30%): Strong regularization, for small adaptation datasets
- Higher rates (>0.5): Typically too aggressive, may harm performance

When to use LoRA-drop over standard LoRA:

- You have limited adaptation data (risk of overfitting)
- You need better generalization to unseen data
- You're fine-tuning on a very specific task but need to maintain general capabilities
- You've observed overfitting with standard LoRA

## How It Works

LoRA-drop extends standard LoRA by adding dropout to the LoRA components during training.
During the forward pass in training mode, a random subset of LoRA components are "dropped out"
(set to zero), forcing the model to learn more robust adaptations that don't rely on any
single component.

Key differences from standard LoRA:

- Applies dropout to LoRA output during training
- Scales LoRA output by (1 - dropout_rate) during inference
- Improves generalization and reduces overfitting
- Particularly useful when adaptation data is limited

