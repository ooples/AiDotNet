---
title: "IGradientComputable<T, TInput, TOutput>"
description: "Base interface for models that can compute gradients explicitly without updating parameters."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Base interface for models that can compute gradients explicitly without updating parameters.

## For Beginners

Regular training computes gradients and immediately updates the model in one step.
This interface separates those two operations:

1. `ILossFunction{` - Calculate which direction improves the model (WITHOUT changing it)
2. `Vector{` - Actually update the model using those directions

This separation is crucial when you need to process gradients before applying them,
such as averaging gradients across multiple GPUs in distributed training.

## How It Works

This interface enables models to compute gradients without immediately applying parameter updates.
This is essential for:

- **Distributed Training**: Compute local gradients, synchronize across workers, then apply averaged gradients
- **Meta-Learning**: Compute gradients on query sets after adaptation (see `ISecondOrderGradientComputable`)
- **Custom Optimization**: Manually control when and how to apply gradients
- **Gradient Analysis**: Inspect gradient values for debugging or monitoring

**Distributed Training Use Case:**
In Data Parallel training (DDP), each GPU:

1. Computes gradients on its local data batch
2. Communicates gradients with other GPUs to compute the average
3. Applies the averaged gradients to update parameters

Without this interface, step 2 would be impossible because gradients would already
be applied in step 1.

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` | Applies pre-computed gradients to update the model parameters. |
| `ComputeGradients(,,ILossFunction<>)` | Computes gradients of the loss function with respect to model parameters for the given data, WITHOUT updating the model parameters. |

