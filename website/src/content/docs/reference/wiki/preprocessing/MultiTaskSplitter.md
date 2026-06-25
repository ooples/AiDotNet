---
title: "MultiTaskSplitter<T>"
description: "Multi-task splitter that ensures consistent splits across multiple related tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.DomainSpecific`

Multi-task splitter that ensures consistent splits across multiple related tasks.

## For Beginners

In multi-task learning, we train a model on multiple related tasks
simultaneously. It's important that the same samples appear in train/test consistently
across all tasks to fairly evaluate transfer learning benefits.

## How It Works

**Example:**
For a model predicting both "sentiment" and "topic" from text:

- Same documents should be in training for both tasks
- Same documents should be in testing for both tasks

**When to Use:**

- Multi-task neural networks
- Transfer learning experiments
- Multi-output regression/classification

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MultiTaskSplitter(Double,Int32,Boolean,Boolean,Int32)` | Creates a new multi-task splitter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Split(Matrix<>,Vector<>)` |  |

