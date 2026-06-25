---
title: "TIMOptions<T, TInput, TOutput>"
description: "Configuration options for TIM (Transductive Information Maximization) (Boudiaf et al., NeurIPS 2020)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for TIM (Transductive Information Maximization) (Boudiaf et al., NeurIPS 2020).

## For Beginners

TIM uses query examples to help classify each other:

**Inductive vs Transductive:**

- Inductive: Classify each query example independently
- Transductive: Use ALL query examples together (they provide info about each other)

**How TIM works:**

1. Compute initial prototypes from support set (like ProtoNets)
2. For each query, compute soft assignments to classes
3. Iteratively refine assignments by maximizing mutual information:
- Conditional entropy: Each query should be confidently assigned to one class
- Marginal entropy: Classes should have balanced assignments overall
- KL divergence: Soft assignments shouldn't deviate too far from initial predictions

**Analogy:**
Imagine sorting a pile of photos into groups:

- Inductive: Sort each photo individually
- TIM: Sort all photos simultaneously, using the fact that groups should be balanced

and each photo should clearly belong to one group

## How It Works

TIM is a transductive method that maximizes mutual information between query features
and their predicted labels. By using ALL query examples jointly (not independently),
it exploits the structure of the query set for better classification.

Reference: Boudiaf, M., Ziko, I., Rony, J., Dolz, J., Piantanida, P., & Ben Ayed, I. (2020).
Information Maximization for Few-Shot Learning. NeurIPS 2020.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TIMOptions(IFullModel<,,>)` | Initializes a new instance of TIMOptions. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` |  |
| `CheckpointFrequency` |  |
| `ConditionalEntropyWeight` | Gets or sets the weight for the conditional entropy term. |
| `DataLoader` | Gets or sets the episodic data loader. |
| `EnableCheckpointing` |  |
| `EvaluationFrequency` |  |
| `EvaluationTasks` |  |
| `GradientClipThreshold` |  |
| `InnerLearningRate` |  |
| `InnerOptimizer` | Gets or sets the inner loop optimizer. |
| `LossFunction` | Gets or sets the loss function. |
| `MarginalEntropyWeight` | Gets or sets the weight for the marginal entropy term. |
| `MetaBatchSize` |  |
| `MetaModel` | Gets or sets the feature extractor model. |
| `MetaOptimizer` | Gets or sets the outer loop optimizer. |
| `NumMetaIterations` |  |
| `OuterLearningRate` |  |
| `RandomSeed` | Gets or sets the random seed. |
| `Temperature` | Gets or sets the temperature for softmax computation. |
| `TransductiveIterations` | Gets or sets the number of transductive refinement iterations. |
| `UseFirstOrder` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `IsValid` |  |

