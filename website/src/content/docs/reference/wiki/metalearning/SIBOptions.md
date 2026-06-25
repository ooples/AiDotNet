---
title: "SIBOptions<T, TInput, TOutput>"
description: "Configuration options for SIB (Sequential Information Bottleneck) (Hu et al., 2020) few-shot learning."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for SIB (Sequential Information Bottleneck) (Hu et al., 2020) few-shot learning.

## For Beginners

SIB balances two competing goals:

**The Information Bottleneck principle:**

1. Keep USEFUL information: Cluster assignments should predict labels well
2. Remove USELESS information: Don't memorize irrelevant details

**How SIB works for few-shot learning:**

1. Initialize cluster centroids from support class prototypes
2. Assign ALL examples (support + query) to clusters
3. Iteratively refine assignments by:
- Reassigning each example to the most informative cluster
- Updating cluster centroids based on new assignments
- Balancing information retention vs. compression

**Analogy: Efficient note-taking**
Imagine taking notes from a lecture:

- Too much detail = you copy everything (no compression, hard to review)
- Too little detail = you miss key points (too much compression)
- SIB finds the sweet spot: keep important info, discard noise

**Key property:** Transductive - uses query set structure for better predictions

## How It Works

SIB uses the information bottleneck principle for transductive few-shot learning.
It iteratively optimizes cluster assignments by maximizing mutual information between
data representations and cluster labels while compressing nuisance information.

Reference: Hu, Y., Gripon, V., & Pateux, S. (2020).
Leveraging the Feature Distribution in Transfer-based Few-Shot Learning.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SIBOptions(IFullModel<,,>)` | Initializes a new instance of SIBOptions. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` |  |
| `Beta` | Gets or sets the compression parameter (beta) for the information bottleneck. |
| `CheckpointFrequency` |  |
| `DataLoader` | Gets or sets the episodic data loader. |
| `EnableCheckpointing` |  |
| `EvaluationFrequency` |  |
| `EvaluationTasks` |  |
| `GradientClipThreshold` |  |
| `InnerLearningRate` |  |
| `InnerOptimizer` | Gets or sets the inner loop optimizer. |
| `LossFunction` | Gets or sets the loss function. |
| `MetaBatchSize` |  |
| `MetaModel` | Gets or sets the feature extractor model. |
| `MetaOptimizer` | Gets or sets the outer loop optimizer. |
| `NumMetaIterations` |  |
| `NumRestarts` | Gets or sets the number of random restarts for avoiding local optima. |
| `NumSIBIterations` | Gets or sets the number of SIB optimization iterations. |
| `OuterLearningRate` |  |
| `RandomSeed` | Gets or sets the random seed. |
| `Temperature` | Gets or sets the temperature for soft cluster assignments. |
| `UseFirstOrder` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `IsValid` |  |

