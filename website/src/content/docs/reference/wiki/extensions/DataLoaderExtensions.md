---
title: "DataLoaderExtensions"
description: "Provides extension methods for data loaders to enhance batch iteration capabilities."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Extensions`

Provides extension methods for data loaders to enhance batch iteration capabilities.

## For Beginners

Extension methods add new capabilities to existing types.
These methods make it easy to iterate through your data in batches with a clean syntax:

## How It Works

These extension methods provide a fluent API for batch iteration, enabling
PyTorch-style and TensorFlow-style data loading patterns.

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateBatches(IBatchIterable<>,Nullable<Int32>)` | Creates a batch configuration builder for fluent batch iteration. |
| `CreateBatchesAsync(IBatchIterable<>,Nullable<Int32>,Int32)` | Creates an async batch configuration builder for fluent async batch iteration. |

