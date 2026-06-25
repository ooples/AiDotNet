---
title: "IContinualLearnerConfig<T>"
description: "Configuration interface for continual learning algorithms."
section: "API Reference"
---

`Interfaces` · `AiDotNet.ContinualLearning.Interfaces`

Configuration interface for continual learning algorithms.

## For Beginners

This interface defines the settings needed for continual learning,
such as learning rates, memory constraints, and regularization parameters.

## How It Works

**Continual Learning** is the ability to learn new tasks sequentially without
forgetting previously learned knowledge. This is challenging because neural networks
tend to suffer from "catastrophic forgetting" - learning new information overwrites
old knowledge.

**Common Strategies Include:**

- **EWC (Elastic Weight Consolidation):** Protects important weights from changing
- **LwF (Learning without Forgetting):** Uses knowledge distillation from teacher model
- **GEM (Gradient Episodic Memory):** Projects gradients to prevent forgetting
- **SI (Synaptic Intelligence):** Tracks online importance of weights
- **Experience Replay:** Stores and replays examples from previous tasks

**Reference:** Parisi et al. "Continual Lifelong Learning with Neural Networks: A Review" (2019)

## Properties

| Property | Summary |
|:-----|:--------|
| `AGemMargin` | Margin for A-GEM (Averaged GEM). |
| `AGemReferenceGradients` | Number of reference gradients for A-GEM. |
| `BatchSize` | Batch size for training. |
| `BiCValidationFraction` | Validation set fraction for BiC bias correction. |
| `ComputeBackwardTransfer` | Compute backward transfer metric. |
| `ComputeForwardTransfer` | Compute forward transfer metric. |
| `DistillationTemperature` | Temperature for knowledge distillation softmax. |
| `DistillationWeight` | Weight for distillation loss vs task loss. |
| `EpochsPerTask` | Number of training epochs per task. |
| `EvaluationFrequency` | Evaluation frequency (every N epochs). |
| `EwcLambda` | EWC regularization strength (lambda). |
| `FisherSamples` | Number of samples for Fisher Information computation. |
| `GemMemoryStrength` | Memory strength for GEM constraint. |
| `GradientClipNorm` | Gradient clipping max norm. |
| `HatSmax` | Smax value for gradient-based attention. |
| `HatSparsity` | Sparsity coefficient for HAT. |
| `ICarlExemplarsPerClass` | Number of exemplars per class for iCaRL. |
| `ICarlUseHerding` | Use herding for exemplar selection. |
| `LearningRate` | Learning rate for training. |
| `MasLambda` | MAS regularization coefficient (lambda). |
| `MaxTasks` | Maximum number of tasks to support. |
| `MemorySize` | Maximum number of examples to store from previous tasks. |
| `MemoryStrategy` | Memory sampling strategy. |
| `NormalizeFisher` | Normalize Fisher Information matrix. |
| `PackNetPruneRatio` | Pruning ratio for PackNet. |
| `PackNetRetrainEpochs` | Retrain epochs after pruning. |
| `PnnLateralScaling` | Lateral connection scaling factor. |
| `PnnUseLateralConnections` | Use lateral connections in progressive networks. |
| `RandomSeed` | Random seed for reproducibility. |
| `SamplesPerTask` | Number of samples per task to store in memory. |
| `SiC` | SI regularization coefficient (c). |
| `SiXi` | SI dampening factor (xi). |
| `UseEmpiricalFisher` | Use empirical Fisher (gradient squared) vs true Fisher. |
| `UseGradientClipping` | Enable gradient clipping. |
| `UsePrioritizedReplay` | Use prioritized experience replay based on sample importance. |
| `UseSoftTargets` | Use soft targets from teacher model. |
| `UseWeightDecay` | Enable weight decay regularization. |
| `WeightDecay` | Weight decay coefficient. |

## Methods

| Method | Summary |
|:-----|:--------|
| `IsValid` | Validates the configuration. |

