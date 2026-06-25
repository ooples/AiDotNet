---
title: "ActiveLearner<T, TInput, TOutput>"
description: "Core implementation of the active learner that orchestrates the active learning loop."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning.Core`

Core implementation of the active learner that orchestrates the active learning loop.

## For Beginners

The ActiveLearner is the main orchestrator that runs the active
learning process. It manages the labeled and unlabeled pools, coordinates with the query
strategy to select informative samples, and trains the model iteratively.

## How It Works

**Active Learning Workflow:**

**Example Usage:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ActiveLearner(IFullModel<,,>,IQueryStrategy<,,>,ActiveLearnerConfig<>)` | Initializes a new ActiveLearner with the specified model, strategy, and configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Configuration` |  |
| `IterationsCompleted` |  |
| `LabeledPool` |  |
| `Model` |  |
| `QueryStrategy` |  |
| `TotalQueries` |  |
| `UnlabeledPool` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddLabeledSamples(Int32[],[])` |  |
| `Evaluate(IDataset<,,>)` |  |
| `GetLearningCurve` |  |
| `Initialize(IDataset<,,>,IDataset<,,>)` |  |
| `Run(IOracle<,>,IStoppingCriterion<>)` |  |
| `RunIteration(IOracle<,>)` |  |
| `SelectNextBatch` |  |
| `TrainModel` |  |

## Events

| Event | Summary |
|:-----|:--------|
| `IterationCompleted` |  |
| `LearningCompleted` |  |
| `SamplesSelected` |  |

