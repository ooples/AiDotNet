---
title: "FineTuningBase<T, TInput, TOutput>"
description: "Base class for all fine-tuning methods."
section: "API Reference"
---

`Base Classes` · `AiDotNet.FineTuning`

Base class for all fine-tuning methods.

## For Beginners

This is the foundation that all specific fine-tuning methods
(like DPO, RLHF, SimPO) build upon. It provides common functionality so each method
only needs to implement its unique training logic.

## How It Works

This abstract class provides common functionality shared by all fine-tuning implementations,
including serialization, option management, and utility methods.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FineTuningBase(FineTuningOptions<>)` | Initializes a new instance of the fine-tuning base class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Category` |  |
| `MethodName` |  |
| `RequiresReferenceModel` |  |
| `RequiresRewardModel` |  |
| `SupportsPEFT` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeArrayLogProbability(Array,Array)` | Computes log probability for array outputs (probability distributions or embeddings). |
| `ComputeEnumerableLogProbability(Object[],Object[])` | Computes log probability for IEnumerable outputs. |
| `ComputeKLDivergence(Double[],Double[])` | Computes the KL divergence between two probability distributions. |
| `ComputeLogProbability(IFullModel<,,>,,)` | Computes the log probability of an output given an input. |
| `ComputeLogProbabilityFromPrediction(,)` | Computes log probability from a prediction and target output. |
| `ComputeScalarLogProbability(Double,Double)` | Computes log probability for scalar numeric outputs. |
| `ComputeSequenceMatchLogProbability(Object[],Object[])` | Computes log probability using sequence element matching for non-numeric objects. |
| `ComputeStringLogProbability(String,String)` | Computes log probability for string outputs using character-level matching. |
| `CreateBatches(FineTuningData<,,>,Int32,Boolean)` | Creates batches from the training data. |
| `Deserialize(Byte[])` |  |
| `EvaluateAsync(IFullModel<,,>,FineTuningData<,,>,CancellationToken)` |  |
| `FineTuneAsync(IFullModel<,,>,FineTuningData<,,>,CancellationToken)` |  |
| `GetDynamicShapeInfo` |  |
| `GetInputShape` |  |
| `GetOptions` |  |
| `GetOutputShape` |  |
| `IsNumericType(Type)` | Checks if a type is a numeric type. |
| `LoadModel(String)` |  |
| `LogProgress(Int32,Int32,Double,String)` | Logs training progress. |
| `LogSigmoid(Double)` | Applies the log sigmoid function. |
| `Reset` |  |
| `SaveModel(String)` |  |
| `Serialize` |  |
| `Sigmoid(Double)` | Applies the sigmoid function. |
| `UpdateMetrics(Double,Int32)` | Updates training metrics with batch results. |
| `ValidateTrainingData(FineTuningData<,,>)` | Validates that the training data is appropriate for this fine-tuning method. |

## Fields

| Field | Summary |
|:-----|:--------|
| `CurrentMetrics` | Metrics collected during training. |
| `NumOps` | The numeric operations helper for type T. |
| `Options` | The configuration options for this fine-tuning method. |
| `Random` | Random number generator for training. |

