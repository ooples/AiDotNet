---
title: "DeepHit<T>"
description: "DeepHit: A deep learning approach to survival analysis with competing risks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Regression`

DeepHit: A deep learning approach to survival analysis with competing risks.

## For Beginners

Unlike DeepSurv (which assumes factors affect risk proportionally at all times),
DeepHit learns the actual probability of an event at each specific time point. This is useful when:

- Risk factors affect survival differently at different times
- You want to predict exact probabilities at specific time horizons
- You have competing risks (multiple ways an event can happen)

Example: "What's the probability a patient experiences disease recurrence (risk 1) vs side effects (risk 2)
within 1 year, 2 years, or 5 years?"

Key concepts:

- Time bins: The time axis is divided into discrete bins (e.g., months 0-12, 12-24, 24-36...)
- PMF: Probability Mass Function - probability of event at each time bin
- CIF: Cumulative Incidence Function - probability of event by time t
- Survival: Probability of no event by time t

## How It Works

DeepHit directly learns the distribution of survival times without making the proportional
hazards assumption. It outputs the probability mass function (PMF) of event times across
discrete time bins and can handle multiple competing risks.

Reference: Lee, C. et al. (2018). "DeepHit: A Deep Learning Approach to Survival Analysis
with Competing Risks". AAAI Conference on Artificial Intelligence.

## Example

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;

double[][] features =
{
    new[] { 1.0, 2.0 }, new[] { 2.0, 3.0 }, new[] { 3.0, 4.0 },
    new[] { 4.0, 5.0 }, new[] { 5.0, 6.0 }, new[] { 6.0, 7.0 }
};
double[] targets = { 3.0, 5.0, 7.0, 9.0, 11.0, 13.0 };

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new DeepHit<double>())
    .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
    .BuildAsync();

Console.WriteLine("Trained DeepHit.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeepHit(DeepHitOptions<>,IRegularization<,Matrix<>,Vector<>>)` | Initializes a new instance of DeepHit. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumberOfTrees` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyLayer(Vector<>[],Matrix<>,Vector<>,Boolean)` | Applies a single layer. |
| `ApplySoftmaxAcrossAll(Vector<>[][])` | Applies softmax across all causes and time bins. |
| `CalculateFeatureImportancesAsync(Int32)` |  |
| `ComputeCIndex(Matrix<>,Vector<>,Vector<>)` | Computes the concordance index (C-index) for model evaluation. |
| `ComputeLossAndGradients(Vector<>[][],Int32[],Vector<>,Int32[])` | Computes loss and gradients. |
| `ComputeTimeDependentAUC(Matrix<>,Vector<>,Vector<>,)` | Computes the time-dependent AUC at specific time horizons. |
| `ConvertTimesToBins(Vector<>)` | Converts times to bin indices. |
| `CreateNewInstance` |  |
| `Deserialize(Byte[])` |  |
| `ForwardPass(Matrix<>,Int32[])` | Forward pass through the network. |
| `GetModelMetadata` |  |
| `GetTimeBinCenter(Int32)` | Gets the center time of a bin as double (for evaluation metrics). |
| `GetTimeBinCenterT(Int32)` | Gets the center time of a bin as T. |
| `GetTimeBinIndex()` | Gets the bin index for a given time. |
| `InitializeBiases(Int32)` | Initializes bias vector with zeros. |
| `InitializeNetwork` | Initializes the neural network architecture. |
| `InitializeTimeBins(Vector<>)` | Initializes time bin edges based on observed times. |
| `InitializeWeights(Int32,Int32)` | Initializes weight matrix with He initialization. |
| `PredictAsync(Matrix<>)` |  |
| `PredictCIF(Matrix<>,Vector<>,Int32)` | Predicts cumulative incidence function (CIF) for a specific risk. |
| `PredictExpectedTime(Matrix<>)` | Predicts expected time to event. |
| `PredictMedianSurvivalTime(Matrix<>)` | Predicts median survival time for each sample. |
| `PredictPMF(Matrix<>)` | Predicts the probability mass function (PMF) of event time for each sample. |
| `PredictSurvival(Matrix<>,Vector<>)` | Predicts survival probability S(t) = P(T > t) at specified times. |
| `Serialize` |  |
| `TrainAsync(Matrix<>,Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_causeBiases` | Cause-specific network biases (one list per cause). |
| `_causeWeights` | Cause-specific network weights (one list per cause). |
| `_numFeatures` | Number of features. |
| `_options` | Configuration options. |
| `_outputBiases` | Output layer biases (for each cause). |
| `_outputWeights` | Output layer weights (for each cause, maps to time bins). |
| `_random` | Random number generator. |
| `_sharedBiases` | Shared network biases. |
| `_sharedWeights` | Shared network weights. |
| `_timeBinEdges` | Time bin edges (discretization of time axis). |

