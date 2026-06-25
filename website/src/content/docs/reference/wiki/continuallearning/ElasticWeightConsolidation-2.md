---
title: "ElasticWeightConsolidation<T, TInput, TOutput>"
description: "ElasticWeightConsolidation<T, TInput, TOutput> — Models & Types in AiDotNet.ContinualLearning.Strategies."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ContinualLearning.Strategies`

_No summary documentation available yet._

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ElasticWeightConsolidation(ILossFunction<>,,Int32)` | Initializes a new EWC strategy with default options. |
| `ElasticWeightConsolidation(ILossFunction<>,EWCOptions<>)` | Initializes a new EWC strategy with custom options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AccumulatedFisher` | Gets the accumulated Fisher Information (for online EWC). |
| `ConsolidatedParameters` | Gets the consolidated parameters (for online EWC). |
| `FisherInformation` | Gets the stored Fisher Information from previous tasks. |
| `Lambda` | Gets the regularization strength (lambda). |
| `MemoryUsageBytes` |  |
| `ModifiesArchitecture` |  |
| `Name` |  |
| `OptimalParameters` | Gets the stored optimal parameters from previous tasks. |
| `RequiresMemoryBuffer` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AdjustGradients(Vector<>)` |  |
| `ComputeFisherInformation(IFullModel<,,>)` | Computes the diagonal Fisher Information Matrix from cached gradients. |
| `ComputeMax(Vector<>)` | Computes the maximum value in a vector. |
| `ComputeMean(Vector<>)` | Computes the mean of a vector. |
| `ComputeOnlineEWCLoss(IFullModel<,,>)` | Computes EWC loss using online formulation (accumulated Fisher). |
| `ComputeOriginalEWCLoss(IFullModel<,,>)` | Computes EWC loss using the original formulation (separate Fisher per task). |
| `ComputeRegularizationLoss(IFullModel<,,>)` |  |
| `FinalizeTask(IFullModel<,,>)` |  |
| `GetStateForSerialization` |  |
| `NormalizeFisherInformation(Vector<>)` | Normalizes Fisher Information to prevent numerical issues. |
| `PrepareForTask(IFullModel<,,>,IDataset<,,>)` |  |
| `Reset` |  |
| `UpdateOnlineEWC(Vector<>,Vector<>)` | Updates the online EWC state with new task information. |

