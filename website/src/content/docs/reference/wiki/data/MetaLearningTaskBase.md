---
title: "MetaLearningTaskBase<T, TInput, TOutput>"
description: "Abstract base class for meta-learning tasks, providing common functionality and validation."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Data.Structures`

Abstract base class for meta-learning tasks, providing common functionality and validation.

## How It Works

This base class implements the `IMetaLearningTask` interface
and provides common functionality for all meta-learning task implementations. It includes
validation, default values, and helper methods to reduce code duplication across concrete
implementations.

Concrete implementations should inherit from this class and provide any additional
functionality specific to their meta-learning scenario (e.g., episode tracking,
hierarchical relationships, continual learning).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MetaLearningTaskBase(Int32,Int32,Int32,String,Dictionary<String,Object>)` | Initializes a new instance of the MetaLearningTaskBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Metadata` | Gets the additional metadata about the task. |
| `Name` | Gets or sets an optional name or identifier for the task. |
| `NumQueryPerClass` | Gets the number of query examples per class. |
| `NumShots` | Gets the number of shots (examples per class) in the support set. |
| `NumWays` | Gets the number of ways (classes) in this task. |
| `QueryInput` | Gets or sets the input features for the query set. |
| `QueryOutput` | Gets or sets the target labels for the query set. |
| `QuerySetX` | Gets the input features for the query set (alias for QueryInput). |
| `QuerySetY` | Gets the target labels for the query set (alias for QueryOutput). |
| `SupportInput` | Gets or sets the input features for the support set. |
| `SupportOutput` | Gets or sets the target labels for the support set. |
| `SupportSetX` | Gets the input features for the support set (alias for SupportInput). |
| `SupportSetY` | Gets the target labels for the support set (alias for SupportOutput). |
| `TaskId` | Gets or sets an optional identifier for the task. |

## Methods

| Method | Summary |
|:-----|:--------|
| `SetMetadata(String,Object)` | Adds or updates a metadata entry. |
| `ToString` | Creates a string representation of the task. |
| `TryGetMetadata(String,)` | Gets a metadata value if it exists. |
| `Validate` | Validates that the task has all required data populated. |

