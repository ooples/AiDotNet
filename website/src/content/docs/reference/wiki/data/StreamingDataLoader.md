---
title: "StreamingDataLoader<T, TInput, TOutput>"
description: "A data loader that streams data from disk or other sources without loading all data into memory."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Loaders`

A data loader that streams data from disk or other sources without loading all data into memory.

## For Beginners

When your dataset is too large to fit in RAM (e.g., millions
of images or text documents), you can't load it all at once. StreamingDataLoader solves
this by reading data piece by piece as needed.

Example:

## How It Works

StreamingDataLoader is designed for datasets that don't fit in memory. Instead of loading
all data upfront, it reads data on-demand from a source, processes it, and yields batches.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StreamingDataLoader(Int32,Func<Int32,CancellationToken,Task<ValueTuple<,>>>,Int32,String,Int32,Int32)` | Initializes a new instance of the StreamingDataLoader class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SampleCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ReadSampleAsync(Int32,CancellationToken)` |  |

