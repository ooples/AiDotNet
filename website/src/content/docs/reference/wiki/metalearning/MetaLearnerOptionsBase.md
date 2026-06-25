---
title: "MetaLearnerOptionsBase<T>"
description: "Base implementation of IMetaLearnerOptions with industry-standard defaults."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning`

Base implementation of IMetaLearnerOptions with industry-standard defaults.

## For Beginners

You can use this class directly or extend it to create
algorithm-specific options classes. The defaults work well for most scenarios.

## How It Works

This class provides sensible defaults for meta-learning configurations based on
established practices from the meta-learning literature (MAML, Reptile, etc.).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MetaLearnerOptionsBase` | Initializes a new instance with default values. |
| `MetaLearnerOptionsBase(IMetaLearnerOptions<>)` | Initializes a new instance by copying values from another options instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` |  |
| `CheckpointFrequency` |  |
| `EnableCheckpointing` |  |
| `EvaluationFrequency` |  |
| `EvaluationTasks` |  |
| `GradientClipThreshold` |  |
| `InnerLearningRate` |  |
| `MetaBatchSize` |  |
| `NumMetaIterations` |  |
| `OuterLearningRate` |  |
| `RandomSeed` |  |
| `UseFirstOrder` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `CreateBuilder` | Creates a builder for fluent configuration. |
| `IsValid` |  |

