---
title: "IActiveLearner<T, TInput, TOutput>"
description: "Interface for active learners."
section: "API Reference"
---

`Interfaces` · `AiDotNet.ActiveLearning.Interfaces`

Interface for active learners.

## For Beginners

An active learner is a machine learning system that
actively selects which data points should be labeled. Instead of labeling all data,
it strategically chooses the most informative samples to learn from.

## How It Works

**Active Learning Workflow:**

**When to Use Active Learning:**

## Properties

| Property | Summary |
|:-----|:--------|
| `Configuration` | Gets the configuration for this active learner. |
| `IterationsCompleted` | Gets the number of active learning iterations completed. |
| `LabeledPool` | Gets the current labeled dataset. |
| `Model` | Gets the underlying model being trained. |
| `QueryStrategy` | Gets the query strategy used for sample selection. |
| `TotalQueries` | Gets the number of queries (labeling requests) made so far. |
| `UnlabeledPool` | Gets the current unlabeled pool. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddLabeledSamples(Int32[],[])` | Adds newly labeled samples to the training set. |
| `Evaluate(IDataset<,,>)` | Evaluates the model on a test dataset. |
| `GetLearningCurve` | Gets the learning curve (performance vs. |
| `Initialize(IDataset<,,>,IDataset<,,>)` | Initializes the active learner with labeled and unlabeled data. |
| `Run(IOracle<,>,IStoppingCriterion<>)` | Runs active learning until a stopping criterion is met. |
| `RunIteration(IOracle<,>)` | Runs a single iteration of active learning. |
| `SelectNextBatch` | Selects the next batch of samples to query. |
| `TrainModel` | Trains the model on the current labeled dataset. |

## Events

| Event | Summary |
|:-----|:--------|
| `IterationCompleted` | Event raised when an iteration is completed. |
| `LearningCompleted` | Event raised when the learning process is complete. |
| `SamplesSelected` | Event raised when samples are selected for labeling. |

