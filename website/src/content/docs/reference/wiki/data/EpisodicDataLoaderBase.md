---
title: "EpisodicDataLoaderBase<T, TInput, TOutput>"
description: "Provides a base implementation for episodic data loaders with common functionality for N-way K-shot meta-learning."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Data.Loaders`

Provides a base implementation for episodic data loaders with common functionality for N-way K-shot meta-learning.

## For Beginners

This is the foundation that all episodic data loaders build upon.

Think of it like a template for creating meta-learning tasks:

- It handles common tasks (validating inputs, organizing data by class, checking requirements)
- Specific loaders just implement how they sample tasks
- This ensures all episodic loaders work consistently and follow SOLID principles

The base class takes care of:

- Validating that your dataset has enough classes and examples
- Building an efficient index (class to example indices) for fast sampling
- Storing configuration (N-way, K-shot, query shots)
- Providing protected access to the dataset and configuration

## How It Works

This abstract class implements the IEpisodicDataLoader interface and provides common functionality
for episodic task sampling. It handles dataset validation, class-to-indices preprocessing, and
parameter validation while allowing derived classes to focus on implementing specific sampling strategies.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EpisodicDataLoaderBase(Matrix<>,Vector<>,Int32,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of the EpisodicDataLoaderBase class with industry-standard defaults. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AvailableClasses` |  |
| `BatchSize` |  |
| `Description` |  |
| `HasNext` |  |
| `KShot` |  |
| `NWay` |  |
| `Name` |  |
| `QueryShots` |  |
| `TotalCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildClassIndex(Vector<>)` | Builds a dictionary mapping each class label to its example indices. |
| `BuildMetaLearningTask(List<Vector<>>,List<>,List<Vector<>>,List<>)` | Builds a MetaLearningTask from lists of examples and labels. |
| `ConvertMatrixToInput(Matrix<>)` | Converts a Matrix to TInput type using the same pattern as ConversionsHelper. |
| `ConvertVectorToOutput(Vector<>)` | Converts a Vector to TOutput type using the same pattern as ConversionsHelper. |
| `GetBatches(Nullable<Int32>,Boolean,Boolean,Nullable<Int32>)` |  |
| `GetBatchesAsync(Nullable<Int32>,Boolean,Boolean,Nullable<Int32>,Int32,CancellationToken)` |  |
| `GetNextBatch` |  |
| `GetNextTask` |  |
| `GetNextTaskCore` | Core task sampling logic to be implemented by derived classes. |
| `GetTaskBatch(Int32)` |  |
| `LoadDataCoreAsync(CancellationToken)` |  |
| `OnReset` |  |
| `SetSeed(Int32)` |  |
| `TryGetNextBatch(MetaLearningTask<,,>)` |  |
| `UnloadDataCore` |  |
| `ValidateConfiguration(Int32,Int32,Int32)` | Validates the N-way K-shot configuration. |
| `ValidateDataset(Matrix<>,Vector<>)` | Validates the dataset inputs. |
| `ValidateDatasetSize(Dictionary<Int32,List<Int32>>,Int32,Int32,Int32)` | Validates that the dataset has sufficient classes and examples per class. |

## Fields

| Field | Summary |
|:-----|:--------|
| `ClassToIndices` | Mapping from class label to list of example indices for that class. |
| `DatasetX` | The feature matrix containing all examples. |
| `DatasetY` | The label vector containing class labels for all examples. |
| `NumOps` | Provides mathematical operations for the numeric type T. |
| `RandomInstance` | Random number generator for task sampling. |
| `_availableClasses` | Array of all available class labels in the dataset. |
| `_kShot` | The number of support examples per class (K in K-shot). |
| `_nWay` | The number of classes per task (N in N-way). |
| `_queryShots` | The number of query examples per class. |
| `_randomLock` | Lock object for thread-safe access to RandomInstance during batch generation. |

