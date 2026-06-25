---
title: "EnsembleTeacherModel<T>"
description: "Ensemble teacher model that combines predictions from multiple teacher models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.KnowledgeDistillation.Teachers`

Ensemble teacher model that combines predictions from multiple teacher models.

## For Beginners

Ensemble learning combines multiple models to create a stronger,
more robust teacher. The intuition is similar to seeking advice from multiple experts rather
than relying on a single expert.

## How It Works

**Benefits of Ensemble Teachers:**

- **Higher Accuracy**: Ensemble outperforms individual models
- **Better Calibration**: Averaging reduces overconfidence
- **Robustness**: Less sensitive to individual model biases
- **Knowledge Diversity**: Student learns from complementary perspectives

**Common Ensemble Strategies:**

- **Uniform Average**: Equal weight to all teachers (default)
- **Weighted Average**: More weight to better-performing teachers
- **Voting**: For classification, majority vote
- **Stacking**: Meta-model combines predictions

**Real-world Analogy:**
Imagine learning to play chess from multiple grandmasters. Each has different playing styles
and strategies. By learning from all of them, you develop a more well-rounded understanding
of the game than you would from just one teacher.

**Practical Example:**
Train 3-5 models with different:

- Initializations (different random seeds)
- Architectures (CNN, ResNet, Transformer)
- Hyperparameters (learning rates, depths)

Combine them to create a powerful ensemble teacher.

**References:**

- You et al. (2017). Learning from Multiple Teacher Networks. KDD.
- Fukuda et al. (2017). Efficient Knowledge Distillation from an Ensemble of Teachers.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EnsembleTeacherModel(ITeacherModel<Vector<>,Vector<>>[],Double[],EnsembleAggregationMode)` | Initializes a new instance of the EnsembleTeacherModel class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `OutputDimension` | Gets the output dimension (same for all teachers). |
| `TeacherCount` | Gets the number of teachers in the ensemble. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateLogits(Vector<>[])` | Aggregates logits from multiple teachers according to the aggregation mode. |
| `GetLogits(Vector<>)` | Gets ensemble logits by combining predictions from all teachers. |
| `LogSoftmax(Vector<>)` | Applies log-softmax to convert logits to log-probabilities. |
| `Softmax(Vector<>)` | Applies softmax to convert logits to probabilities. |
| `UpdateWeights(Vector<>[],Vector<>[])` | Updates teacher weights based on performance (for adaptive weighting). |

