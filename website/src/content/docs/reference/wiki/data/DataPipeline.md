---
title: "DataPipeline<T>"
description: "Provides TensorFlow-style data pipeline operations for transforming and processing data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Pipeline`

Provides TensorFlow-style data pipeline operations for transforming and processing data.

## For Beginners

Data pipelines let you build complex data processing
workflows step by step. Each operation transforms the data in some way:

## How It Works

DataPipeline provides a fluent API for building data processing pipelines similar to
TensorFlow's tf.data API. Operations are lazily evaluated and can be chained together.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DataPipeline(Func<IEnumerable<>>)` | Initializes a new instance of the DataPipeline class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SourceFactory` | Gets the source factory for this pipeline. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Batch(Int32,Boolean)` | Groups elements into batches. |
| `Cache` | Caches elements in memory for faster subsequent iterations. |
| `Concat(DataPipeline<>)` | Concatenates another pipeline to this one. |
| `Filter(Func<,Boolean>)` | Filters elements based on a predicate. |
| `Flatten(Func<,IEnumerable<>>)` | Flattens a pipeline by extracting enumerables from each element. |
| `ForEach(Action<>)` | Applies a side effect to each element without modifying it. |
| `From(IBatchIterable<>)` | Creates a new DataPipeline from a batch iterable. |
| `From(IEnumerable<>)` | Creates a new DataPipeline from an enumerable source. |
| `GetEnumerator` |  |
| `Interleave(DataPipeline<>[])` | Interleaves multiple pipelines. |
| `Map(Func<,>)` | Applies a transformation function to each element. |
| `MapAsync(Func<,CancellationToken,Task<>>,Int32)` | Applies an async transformation function to each element. |
| `Prefetch(Int32)` | Prefetches elements in the background for improved performance. |
| `Repeat(Nullable<Int32>)` | Repeats the pipeline indefinitely or a specified number of times. |
| `Shuffle(Int32,Nullable<Int32>)` | Shuffles elements using a buffer. |
| `Skip(Int32)` | Skips the first N elements. |
| `Take(Int32)` | Takes only the first N elements. |
| `Zip(DataPipeline<>)` | Zips this pipeline with another to create pairs. |

