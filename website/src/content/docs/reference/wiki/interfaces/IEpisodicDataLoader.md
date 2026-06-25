---
title: "IEpisodicDataLoader<T, TInput, TOutput>"
description: "Interface for data loaders that provide episodic tasks for meta-learning."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for data loaders that provide episodic tasks for meta-learning.

## For Beginners

Meta-learning is "learning to learn".

**Standard ML**: Train on lots of cat/dog images, then classify new cat/dog images.

**Meta-learning**: Train on many different tasks (cats vs dogs, cars vs planes, etc.),
then when given a *new* task with only a few examples, quickly learn to do it.

**N-way K-shot** means:

- **N-way**: Each task has N different classes to distinguish
- **K-shot**: You get K examples of each class to learn from

**Example: 5-way 1-shot**

- Given 5 new animal types you've never seen
- With only 1 example image of each
- Classify new images into one of these 5 types

The episodic data loader creates these mini-tasks for training.

## How It Works

This interface is for meta-learning scenarios using N-way K-shot learning,
where the loader generates tasks consisting of:

- Support set: K examples per class for N classes (used to adapt the model)
- Query set: Additional examples for evaluation after adaptation

## Properties

| Property | Summary |
|:-----|:--------|
| `AvailableClasses` | Gets the total number of available classes in the dataset. |
| `KShot` | Gets the number of support examples per class (K in K-shot). |
| `NWay` | Gets the number of classes per task (N in N-way). |
| `QueryShots` | Gets the number of query examples per class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetNextTask` | Gets the next meta-learning task (support set + query set). |
| `GetTaskBatch(Int32)` | Gets multiple meta-learning tasks as a batch. |
| `SetSeed(Int32)` | Sets the random seed for reproducible task sampling. |

