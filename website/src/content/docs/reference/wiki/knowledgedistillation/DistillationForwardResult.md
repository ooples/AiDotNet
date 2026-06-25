---
title: "DistillationForwardResult<T>"
description: "Encapsulates the result of a forward pass during knowledge distillation training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.KnowledgeDistillation`

Encapsulates the result of a forward pass during knowledge distillation training.

## For Beginners

When training with knowledge distillation, we need more than just
the final output of a model. We also need intermediate layer activations to enable advanced
distillation techniques (like feature matching or neuron selectivity). This class packages both
the final output and optional intermediate activations together.

## How It Works

**Components:**

- **FinalOutput**: The model's final predictions (e.g., class logits). Shape: [batch_size x num_classes]
- **IntermediateActivations**: Internal layer outputs collected during forward pass (optional)

**Example:**

**Used By:**

- Teacher models: Provide reference outputs and activations
- Student models: Provide outputs to compare against teacher
- Distillation strategies: Compute loss from both final outputs and intermediate activations

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DistillationForwardResult(Matrix<>)` | Initializes a new instance of DistillationForwardResult with final output only. |
| `DistillationForwardResult(Matrix<>,IntermediateActivations<>)` | Initializes a new instance of DistillationForwardResult with final output and intermediate activations. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FinalOutput` | The final output of the model's forward pass. |
| `HasIntermediateActivations` | Checks if intermediate activations are available. |
| `IntermediateActivations` | Optional intermediate layer activations collected during the forward pass. |

