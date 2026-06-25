---
title: "MetaTrainingResult<T>"
description: "Results from a complete meta-training run with history tracking."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Results`

Results from a complete meta-training run with history tracking.

## For Beginners

Meta-training is the process of training your model to be good at
learning new tasks quickly. This happens over many iterations:

1. Sample a batch of tasks
2. Adapt to each task (inner loop)
3. Update meta-parameters based on how well adaptations worked (outer loop)
4. Repeat for many iterations

This result tracks:

- **Learning curves:** How loss and accuracy change over iterations
- **Final performance:** The end results after training
- **Training time:** How long it took
- **Convergence:** Whether training successfully improved the model

Use this to:

- Monitor training progress
- Diagnose training issues
- Compare different meta-learning configurations
- Report results in papers or documentation

## How It Works

This class aggregates metrics across an entire meta-training session, tracking how performance
evolves over many meta-iterations. It combines the functionality of what were previously separate
"Metrics" and "Metadata" classes into a unified Result pattern consistent with the codebase.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MetaTrainingResult(Vector<>,Vector<>,TimeSpan,Dictionary<String,>)` | Initializes a new instance with complete training history. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AccuracyHistory` | Gets the accuracy history across all iterations. |
| `AdditionalMetrics` | Gets algorithm-specific metrics collected during training. |
| `FinalAccuracy` | Gets the final accuracy after training. |
| `FinalLoss` | Gets the final meta-loss after training. |
| `InitialAccuracy` | Gets the initial accuracy before training. |
| `InitialLoss` | Gets the initial meta-loss before training. |
| `LossHistory` | Gets the meta-loss history across all iterations. |
| `TotalIterations` | Gets the total number of meta-training iterations completed. |
| `TrainingTime` | Gets the total time taken for meta-training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateAccuracyImprovement` | Calculates the total improvement in accuracy from start to finish. |
| `CalculateLossImprovement` | Calculates the total improvement in loss from start to finish. |
| `FindBestAccuracy` | Finds the best (highest) accuracy achieved during training. |
| `FindBestLoss` | Finds the best (lowest) loss achieved during training. |
| `GenerateReport` | Generates a comprehensive training report. |
| `HasConverged(Int32,Double)` | Checks if training converged based on loss stabilization. |

