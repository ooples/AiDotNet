---
title: "SGDClassifier<T>"
description: "Stochastic Gradient Descent classifier for large-scale learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Classification.Linear`

Stochastic Gradient Descent classifier for large-scale learning.

## For Beginners

Instead of computing gradients over the entire dataset, SGD:

1. Picks one training sample
2. Computes how wrong the prediction is
3. Updates weights to reduce that error
4. Repeats for all samples (one epoch)
5. Repeats for multiple epochs

Benefits:

- Very fast for large datasets
- Can handle streaming data
- Often finds good solutions quickly

Trade-offs:

- Noisy updates (not always improving)
- Requires tuning learning rate
- May oscillate near optimal solution

## How It Works

SGD is an optimization technique that updates weights using one sample at a time.
This makes it very efficient for large datasets that don't fit in memory.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SGDClassifier(LinearClassifierOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of the SGDClassifier class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `ComputeLossAndGradient(,,,)` | Computes loss and gradient for the specified loss function. |
| `CreateNewInstance` |  |
| `Train(Matrix<>,Vector<>)` | Returns the model type identifier for this classifier. |

