---
title: "ContinualLearnerConfig<T>"
description: "Production-ready configuration for continual learning algorithms."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.ContinualLearning.Config`

Production-ready configuration for continual learning algorithms.

## For Beginners

This class provides all settings needed for continual learning.
All properties have industry-standard defaults set in the constructor.

## How It Works

**Usage Example:**

**Industry Standard Defaults:**

- Learning Rate: 0.001 (Adam optimizer standard)
- Batch Size: 32 (balance of speed and gradient noise)
- EWC Lambda: 1000 (based on Kirkpatrick et al. 2017)
- Distillation Temperature: 2.0 (based on Li and Hoiem 2017)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ContinualLearnerConfig` | Initializes a new instance with industry-standard default values. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AGemMargin` |  |
| `AGemReferenceGradients` |  |
| `BatchSize` |  |
| `BiCValidationFraction` |  |
| `ComputeBackwardTransfer` |  |
| `ComputeForwardTransfer` |  |
| `DistillationTemperature` |  |
| `DistillationWeight` |  |
| `EpochsPerTask` |  |
| `EvaluationFrequency` |  |
| `EwcLambda` |  |
| `FisherSamples` |  |
| `GemMemoryStrength` |  |
| `GradientClipNorm` |  |
| `HatSmax` |  |
| `HatSparsity` |  |
| `ICarlExemplarsPerClass` |  |
| `ICarlUseHerding` |  |
| `LearningRate` |  |
| `MasLambda` |  |
| `MaxTasks` |  |
| `MemorySize` |  |
| `MemoryStrategy` |  |
| `NormalizeFisher` |  |
| `OnlineEwcGamma` |  |
| `PackNetPruneRatio` |  |
| `PackNetRetrainEpochs` |  |
| `PnnLateralScaling` |  |
| `PnnUseLateralConnections` |  |
| `RandomSeed` |  |
| `SamplesPerTask` |  |
| `SiC` |  |
| `SiXi` |  |
| `UseEmpiricalFisher` |  |
| `UseGradientClipping` |  |
| `UsePrioritizedReplay` |  |
| `UseSoftTargets` |  |
| `UseWeightDecay` |  |
| `WeightDecay` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ForEwc(,Nullable<Int32>)` | Creates a configuration optimized for EWC strategy. |
| `ForExperienceReplay(Nullable<Int32>,Nullable<MemorySamplingStrategy>,Nullable<Boolean>)` | Creates a configuration optimized for experience replay. |
| `ForGem(,Nullable<Int32>)` | Creates a configuration optimized for GEM strategy. |
| `ForLwf()` | Creates a configuration optimized for LwF strategy with default distillation weight. |
| `ForLwf(,)` | Creates a configuration optimized for LwF strategy. |
| `IsValid` |  |

