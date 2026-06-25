---
title: "IMetaLearningTask<T, TInput, TOutput>"
description: "Represents a single meta-learning task for few-shot learning."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Represents a single meta-learning task for few-shot learning.

## Properties

| Property | Summary |
|:-----|:--------|
| `Metadata` | Gets the additional metadata about the task. |
| `Name` | Gets an optional name or identifier for the task. |
| `NumQueryPerClass` | Gets the number of query examples per class. |
| `NumShots` | Gets the number of shots (examples per class) in the support set. |
| `NumWays` | Gets the number of ways (classes) in this task. |
| `QueryInput` | Gets the input features for the query set. |
| `QueryOutput` | Gets the target labels for the query set. |
| `QuerySetX` | Gets the input features for the query set (alias for QueryInput). |
| `QuerySetY` | Gets the target labels for the query set (alias for QueryOutput). |
| `SupportInput` | Gets the input features for the support set. |
| `SupportOutput` | Gets the target labels for the support set. |
| `SupportSetX` | Gets the input features for the support set (alias for SupportInput). |
| `SupportSetY` | Gets the target labels for the support set (alias for SupportOutput). |
| `TaskId` | Gets or sets an optional identifier for the task. |

