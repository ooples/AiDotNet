---
title: "DistillationStrategyBase<T>"
description: "Abstract base class for knowledge distillation strategies."
section: "API Reference"
---

`Base Classes` · `AiDotNet.KnowledgeDistillation`

Abstract base class for knowledge distillation strategies.
Provides common functionality for computing losses and gradients in student-teacher training.

## For Beginners

A distillation strategy defines how to measure the difference
between student and teacher predictions. This base class provides common functionality that
all strategies need, like temperature and alpha parameters.

## How It Works

**Design Philosophy:**
Different distillation strategies focus on different aspects:

- **Response-based**: Match final outputs (logits/probabilities)
- **Feature-based**: Match intermediate layer representations
- **Relation-based**: Match relationships between samples
- **Attention-based**: Match attention patterns (for transformers)

This base class ensures all strategies handle temperature and alpha consistently,
while allowing flexibility in how loss is computed.

**Batch Processing:** All strategies now operate on batches (Matrix<T>) for efficiency.
Each row in the matrices represents one sample in the batch.

**Template Method Pattern:** The base class defines the structure (properties, validation),
and subclasses implement the specifics (loss computation logic).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DistillationStrategyBase(Double,Double)` | Initializes a new instance of the distillation strategy base class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Alpha` | Gets or sets the balance parameter between hard loss and soft loss. |
| `Engine` | Hardware-accelerated engine for vector/tensor operations. |
| `Temperature` | Gets or sets the temperature parameter for softening probability distributions. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeGradient(Matrix<>,Matrix<>,Matrix<>)` | Computes the gradient of the distillation loss for backpropagation. |
| `ComputeLoss(Matrix<>,Matrix<>,Matrix<>)` | Computes the distillation loss between student and teacher batch outputs. |
| `ValidateLabelDimensions(Matrix<>,Matrix<>)` | Validates that batch outputs and labels have matching dimensions (if labels are provided). |
| `ValidateOutputDimensions(Matrix<>,Matrix<>)` | Validates that student and teacher batch outputs have matching dimensions. |

## Fields

| Field | Summary |
|:-----|:--------|
| `Epsilon` | Gets the epsilon value for numerical stability (to avoid log(0), division by zero, etc.). |
| `NumOps` | Numeric operations for the specified type T. |

