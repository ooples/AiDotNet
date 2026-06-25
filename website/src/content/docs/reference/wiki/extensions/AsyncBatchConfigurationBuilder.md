---
title: "AsyncBatchConfigurationBuilder<TBatch>"
description: "Builder for configuring async batch iteration with a fluent API."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Extensions`

Builder for configuring async batch iteration with a fluent API.

## How It Works

This builder allows chaining configuration methods before async iteration.
Implements IAsyncEnumerable to support await foreach loops.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AsyncBatchConfigurationBuilder(IBatchIterable<>,Nullable<Int32>,Int32)` | Initializes a new instance of the AsyncBatchConfigurationBuilder. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DropLast` | Drops the last incomplete batch if the dataset doesn't divide evenly. |
| `GetAsyncEnumerator(CancellationToken)` | Returns an async enumerator that iterates through the batches. |
| `KeepLast` | Keeps the last batch even if incomplete. |
| `NoShuffle` | Disables shuffling of data before batching. |
| `Shuffled` | Enables shuffling of data before batching. |
| `WithSeed(Int32)` | Sets a random seed for reproducible shuffling. |

