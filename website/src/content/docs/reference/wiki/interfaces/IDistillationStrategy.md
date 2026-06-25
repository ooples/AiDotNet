---
title: "IDistillationStrategy<T>"
description: "Defines a strategy for computing knowledge distillation loss between student and teacher models."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines a strategy for computing knowledge distillation loss between student and teacher models.

## For Beginners

A distillation strategy determines how to measure the difference
between what the student predicts and what the teacher predicts. Different strategies can
focus on different aspects:

- Response-based: Compare final outputs
- Feature-based: Compare intermediate layer features
- Relation-based: Compare relationships between samples

## How It Works

The most common approach (Hinton et al., 2015) combines two losses:

1. Hard loss: How well the student matches the true labels
2. Soft loss: How well the student mimics the teacher's predictions

This combination allows the student to both get the right answers (hard loss) and
learn the teacher's reasoning (soft loss).

**Batch Processing:** This interface operates on batches (Matrix<T>) for efficiency.
Each row in the matrices represents one sample in the batch.

**Interface Design Note:** This interface uses a single type parameter <T> for
numeric operations. All input/output types are Matrix<T> for batch processing. There is no
second type parameter TOutput - the output type is always Matrix<T> for gradients and T for loss values.

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Gets or sets the balance parameter (alpha) between hard loss and soft loss. |
| `Temperature` | Gets or sets the temperature parameter for softening probability distributions. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeGradient(Matrix<>,Matrix<>,Matrix<>)` | Computes the gradient of the distillation loss for backpropagation. |
| `ComputeLoss(Matrix<>,Matrix<>,Matrix<>)` | Computes the distillation loss between student and teacher batch outputs. |

