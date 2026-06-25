---
title: "OpenMAMLOptions<T, TInput, TOutput>"
description: "Configuration options for Open-MAML (open-set MAML for open-world few-shot learning)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for Open-MAML (open-set MAML for open-world few-shot learning).

## For Beginners

Standard few-shot assumes all queries belong to known classes.
Open-MAML handles the realistic case where some queries are from unknown classes:

1. Adapt to support set like MAML
2. Compute prototype distances for each query
3. If a query is too far from ALL prototypes, classify as "unknown"
4. Threshold is learned during meta-training

## How It Works

Open-MAML extends MAML to handle open-set scenarios where query examples may belong
to unknown classes not present in the support set. It adds an outlier detection mechanism
to MAML's adaptation process.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OpenMAMLOptions(IFullModel<,,>)` | Initializes a new instance of OpenMAMLOptions. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` |  |
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
| `OpenSetTaskFraction` | Gets or sets the fraction of tasks that include open-set queries during training. |
| `OpenSetThreshold` | Gets or sets the initial threshold for open-set detection. |
| `OuterLearningRate` |  |
| `RandomSeed` | Gets or sets the random seed. |
| `UseFirstOrder` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `IsValid` |  |

