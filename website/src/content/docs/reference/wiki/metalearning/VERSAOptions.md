---
title: "VERSAOptions<T, TInput, TOutput>"
description: "Configuration options for VERSA (Versatile and Efficient Few-shot Learning) algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for VERSA (Versatile and Efficient Few-shot Learning) algorithm.

## For Beginners

VERSA takes a completely different approach to few-shot learning:

**Traditional meta-learning (MAML, etc.):**
"Here are 5 examples. Let me run gradient descent to learn a classifier..." (slow)

**VERSA:**
"Here are 5 examples. *single forward pass* Here's your classifier." (instant)

How? VERSA trains a separate neural network (the "amortization network") whose job
is to look at a set of examples and immediately output the weights for a classifier.
It's like having a factory that produces customized classifiers on demand.

Key advantages:

- No inner-loop optimization at all (fastest possible adaptation)
- Naturally handles variable numbers of support examples
- The amortization network learns to extract task-relevant statistics

## How It Works

VERSA uses an amortization network to produce task-specific parameters in a single
forward pass, eliminating the need for inner-loop gradient descent entirely.
Given support examples, the amortization network directly outputs classifier parameters.

Reference: Gordon, J., Bronskill, J., Bauer, M., Nowozin, S., & Turner, R. E. (2019).
Meta-Learning Probabilistic Inference for Prediction. ICLR 2019.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VERSAOptions(IFullModel<,,>)` | Initializes a new instance of VERSAOptions. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` |  |
| `AggregationMethod` | Gets or sets the aggregation method for support set features. |
| `AmortizationDropout` | Gets or sets the dropout rate for the amortization network. |
| `AmortizationHiddenDim` | Gets or sets the hidden dimension of the amortization network. |
| `AmortizationNumLayers` | Gets or sets the number of hidden layers in the amortization network. |
| `CheckpointFrequency` |  |
| `DataLoader` | Gets or sets the episodic data loader. |
| `EnableCheckpointing` |  |
| `EvaluationFrequency` |  |
| `EvaluationTasks` |  |
| `GradientClipThreshold` |  |
| `InnerLearningRate` |  |
| `InnerOptimizer` | Gets or sets the inner loop optimizer (unused for VERSA). |
| `LossFunction` | Gets or sets the loss function. |
| `MetaBatchSize` |  |
| `MetaModel` | Gets or sets the feature extractor model. |
| `MetaOptimizer` | Gets or sets the outer loop optimizer. |
| `NumMetaIterations` |  |
| `OuterLearningRate` |  |
| `RandomSeed` | Gets or sets the random seed. |
| `UseFirstOrder` |  |
| `UseProbabilistic` | Gets or sets whether to use a probabilistic (Bayesian) formulation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `IsValid` |  |

