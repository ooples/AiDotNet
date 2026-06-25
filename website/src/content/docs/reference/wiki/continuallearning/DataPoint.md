---
title: "DataPoint<T, TInput, TOutput>"
description: "Represents a single data point with input and output."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ContinualLearning.Memory`

Represents a single data point with input and output.

## For Beginners

A data point represents one example from the training data.
It contains the input (what we show the model), the expected output (what we want it to predict),
and a task ID (which task this example belongs to).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DataPoint(,,Int32)` | Initializes a new data point. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AddedAt` | Gets or sets the timestamp when this data point was added. |
| `Input` | Gets the input data. |
| `Metadata` | Gets or sets additional metadata about this data point. |
| `Output` | Gets the expected output data. |
| `TaskId` | Gets the task identifier this data point belongs to. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ToString` | Returns a string representation of the data point. |

