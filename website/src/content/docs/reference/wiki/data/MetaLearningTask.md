---
title: "MetaLearningTask<T, TInput, TOutput>"
description: "Represents a single meta-learning task for few-shot learning, containing support and query sets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Structures`

Represents a single meta-learning task for few-shot learning, containing support and query sets.

## For Beginners

Meta-learning is about "learning to learn" - training a model to quickly adapt
to new tasks with only a few examples. Think of it like learning a language:

- Traditional learning: Learn one specific language from thousands of examples
- Meta-learning: Learn the general skill of language acquisition so you can learn new languages faster

A MetaLearningTask represents one mini-problem in this process:

- **Support Set:** A few labeled examples the model can study (like a mini training set)
- **Query Set:** Examples to test how well the model adapted (like a mini test set)

For example, in 5-way 3-shot classification:

- The support set has 5 classes with 3 examples each (15 total examples)
- The query set has examples from those same 5 classes to test performance

The model learns from many such tasks, developing the ability to quickly adapt to new tasks.

## How It Works

In meta-learning, particularly few-shot learning, a task is a small classification or regression problem
sampled from a larger dataset. Each task contains a support set (for adapting/learning) and a query set
(for evaluating the adaptation).

**Thread Safety:** This class is not thread-safe. Create separate instances for concurrent access.

**Performance:** This is a lightweight container class with O(1) property access.
Memory usage depends on the size of the tensors stored.

## Properties

| Property | Summary |
|:-----|:--------|
| `Metadata` | Gets or sets the additional metadata about the task. |
| `Name` | Gets or sets an optional name or identifier for the task. |
| `NumQueryPerClass` | Gets or sets the number of query examples per class. |
| `NumShots` | Gets or sets the number of shots (examples per class) in the support set. |
| `NumWays` | Gets or sets the number of ways (classes) in this task. |
| `QueryInput` | Gets the input features for the query set (alias for QuerySetX). |
| `QueryOutput` | Gets the target labels for the query set (alias for QuerySetY). |
| `QuerySetX` | Gets or sets the input features for the query set. |
| `QuerySetY` | Gets or sets the target labels for the query set. |
| `SupportInput` | Gets the input features for the support set (alias for SupportSetX). |
| `SupportOutput` | Gets the target labels for the support set (alias for SupportSetY). |
| `SupportSetX` | Gets or sets the input features for the support set. |
| `SupportSetY` | Gets or sets the target labels for the support set. |
| `TaskId` | Gets or sets an optional task identifier. |

