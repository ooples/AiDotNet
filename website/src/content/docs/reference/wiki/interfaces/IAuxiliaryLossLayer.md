---
title: "IAuxiliaryLossLayer<T>"
description: "Interface for neural network layers that report auxiliary losses in addition to the primary task loss."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for neural network layers that report auxiliary losses in addition to the primary task loss.
Extends `IDiagnosticsProvider<T>` to provide diagnostic information about auxiliary loss computation.

## For Beginners

Think of auxiliary losses as "side goals" for training a neural network.

While the primary loss tells the network "make accurate predictions," auxiliary losses add
additional objectives like:

- "Use all experts equally" (load balancing in Mixture-of-Experts)
- "Keep activations small" (regularization)
- "Learn similar representations" (similarity objectives)

Real-world analogy:
Imagine you're training to be a chef (primary goal: make delicious food). But you also have
auxiliary goals:

- Keep your workspace clean (regularization)
- Use all your tools equally (load balancing)
- Work efficiently (computational constraints)

These auxiliary goals don't directly make the food taste better, but they help you become
a better, more well-rounded chef.

In the training loop, auxiliary losses are typically combined with the primary loss:

Where alpha is a weight that balances the importance of the auxiliary objective.

## How It Works

Auxiliary losses are additional loss terms that help guide training beyond the primary task objective.
They are particularly useful in complex architectures where certain desirable properties (like
balanced resource utilization or regularization) need explicit encouragement during training.

**Common Use Cases:**

- **Load Balancing (MoE):** Encourage balanced expert usage to prevent some experts from being underutilized
- **Sparsity Regularization:** Encourage sparse activations to improve efficiency
- **Contrastive Learning:** Encourage similar inputs to have similar representations
- **Multi-Task Learning:** Additional task objectives that share representations

**Implementation Example:**

## Properties

| Property | Summary |
|:-----|:--------|
| `AuxiliaryLossWeight` | Gets or sets the weight (coefficient) for the auxiliary loss. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeAuxiliaryLoss` | Computes the auxiliary loss for this layer based on the most recent forward pass. |

