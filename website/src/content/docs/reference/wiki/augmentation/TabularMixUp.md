---
title: "TabularMixUp<T>"
description: "Applies MixUp augmentation to tabular data by linearly interpolating between samples."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Tabular`

Applies MixUp augmentation to tabular data by linearly interpolating between samples.

## For Beginners

MixUp creates new training samples by blending two existing
samples together. If you have sample A and sample B, MixUp creates a new sample that's
(λ × A) + ((1-λ) × B), where λ is randomly chosen. The labels are blended the same way.

## How It Works

**Benefits:**

- Regularizes the model by creating "virtual" training examples
- Encourages smooth decision boundaries
- Reduces overconfidence in predictions

**When to use:**

- Classification tasks with numerical features
- When you want to reduce overfitting
- When decision boundaries should be smooth

**Reference:** Zhang et al., "mixup: Beyond Empirical Risk Minimization" (2017)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TabularMixUp(Double,Double)` | Creates a new MixUp augmentation for tabular data. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyAugmentation(Matrix<>,AugmentationContext<>)` |  |
| `MixWithLabels(Matrix<>,Matrix<>,Vector<>,Vector<>,AugmentationContext<>)` | Applies MixUp to two data matrices and their labels, returning mixed results. |

