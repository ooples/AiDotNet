---
title: "BatchConfigurationBuilder<TBatch>"
description: "Builder for configuring batch iteration with a fluent API."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Extensions`

Builder for configuring batch iteration with a fluent API.

## How It Works

This builder allows chaining configuration methods before iteration.
Implements IEnumerable to support foreach loops and LINQ operations.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BatchConfigurationBuilder(IBatchIterable<>,Nullable<Int32>)` | Initializes a new instance of the BatchConfigurationBuilder. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DropLast` | Drops the last incomplete batch if the dataset doesn't divide evenly. |
| `GetEnumerator` | Returns an enumerator that iterates through the batches. |
| `KeepLast` | Keeps the last batch even if incomplete. |
| `NoShuffle` | Disables shuffling of data before batching. |
| `Shuffled` | Enables shuffling of data before batching. |
| `System#Collections#IEnumerable#GetEnumerator` | Returns an enumerator that iterates through the batches. |
| `WithSeed(Int32)` | Sets a random seed for reproducible shuffling. |

