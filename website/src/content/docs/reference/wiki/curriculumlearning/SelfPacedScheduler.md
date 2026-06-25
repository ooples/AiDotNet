---
title: "SelfPacedScheduler<T>"
description: "Self-paced curriculum scheduler that adapts sample selection based on model performance."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CurriculumLearning.Schedulers`

Self-paced curriculum scheduler that adapts sample selection based on model performance.

## For Beginners

Unlike fixed schedulers, self-paced learning dynamically
selects samples based on the model's current ability. Samples with loss below a
threshold are considered "easy enough" and included in training. The threshold
increases over time to include progressively harder samples.

## How It Works

**How It Works:**

**Self-Paced Regularizers:**

**References:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SelfPacedScheduler(Int32,,,,SelfPaceRegularizer)` | Initializes a new instance of the `SelfPacedScheduler` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CurrentThreshold` | Gets the current pace threshold (lambda). |
| `GrowthRate` | Gets or sets the growth rate for the pace parameter. |
| `Name` | Gets the name of this scheduler. |
| `PaceParameter` | Gets or sets the current pace threshold (lambda). |
| `SampleWeights` | Gets sample weights from the last selection. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeHardWeight()` | Hard (binary) self-pace regularizer. |
| `ComputeLinearWeight()` | Linear soft-weighting regularizer. |
| `ComputeLogarithmicWeight()` | Logarithmic regularizer for smoother transitions. |
| `ComputeMixtureWeight()` | Mixture (soft-hard) regularizer. |
| `ComputeSampleWeight()` | Computes the weight for a sample based on its loss and current lambda. |
| `ComputeSampleWeights(Vector<>)` | Computes sample weights based on current losses and pace threshold. |
| `GetCurrentIndices(Int32[],Int32)` | Selects samples based on difficulty scores. |
| `GetDataFraction` | Gets the current data fraction (estimated from lambda progression). |
| `GetStatistics` | Gets scheduler-specific statistics. |
| `Reset` | Resets the scheduler to initial state. |
| `ResolveDefault(,Double)` | Resolves a nullable generic parameter, returning the fallback if the value is null or default(T). |
| `SelectSamplesWithWeights(Vector<>)` | Selects samples based on current losses and pace threshold. |
| `StepEpoch(CurriculumEpochMetrics<>)` | Updates lambda threshold based on epoch metrics. |

