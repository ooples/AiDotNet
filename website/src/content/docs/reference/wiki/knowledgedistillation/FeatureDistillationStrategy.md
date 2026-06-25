---
title: "FeatureDistillationStrategy<T>"
description: "Implements feature-based knowledge distillation (FitNets) where the student learns to match the teacher's intermediate layer representations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.KnowledgeDistillation`

Implements feature-based knowledge distillation (FitNets) where the student learns to match
the teacher's intermediate layer representations.

## For Beginners

While standard distillation transfers knowledge through final outputs,
feature distillation goes deeper by matching intermediate layer activations. This helps the student
learn not just what the teacher predicts, but how it thinks.

## How It Works

**Why Feature Distillation?**

- **Better for Different Architectures**: When student and teacher have very different structures
- **Richer Knowledge Transfer**: Captures hierarchical feature learning
- **Improved Generalization**: Student learns more robust representations
- **Complementary to Response Distillation**: Can be combined with standard distillation

**Real-world Analogy:**
Imagine learning to paint from a master artist. Standard distillation is like copying only the
final painting. Feature distillation is like watching the master's brush strokes, color mixing,
and layering techniques - learning the process, not just the result.

**How It Works:**

1. Extract features from a teacher layer (e.g., conv3 in ResNet)
2. Extract features from corresponding student layer
3. Minimize MSE (Mean Squared Error) between them
4. Optionally use a projection layer if dimensions don't match

**Common Applications:**

- ResNet → MobileNet: Match convolutional feature maps
- BERT → DistilBERT: Match transformer layer outputs
- Teacher and student with different widths/depths

**References:**

- Romero, A., et al. (2014). FitNets: Hints for Thin Deep Nets. arXiv:1412.6550

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FeatureDistillationStrategy(String[],Double)` | Initializes a new instance of the FeatureDistillationStrategy class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultLossFunction` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeFeatureGradient(Vector<>,Vector<>)` | Computes the gradient of feature loss for backpropagation. |
| `ComputeFeatureLoss(Func<String,Vector<>>,Func<String,Vector<>>,Vector<>)` | Computes the feature matching loss between student and teacher intermediate representations. |
| `ComputeMSE(Vector<>,Vector<>)` | Computes Mean Squared Error between two feature vectors. |
| `DeepCopy` |  |
| `GetParameters` |  |
| `Predict(Tensor<>)` |  |
| `SetParameters(Vector<>)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `WithParameters(Vector<>)` |  |

