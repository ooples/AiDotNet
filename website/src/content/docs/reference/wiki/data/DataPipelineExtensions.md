---
title: "DataPipelineExtensions"
description: "Extension methods for creating data pipelines from various sources."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Data.Pipeline`

Extension methods for creating data pipelines from various sources.

## Methods

| Method | Summary |
|:-----|:--------|
| `PaddedBatch(DataPipeline<>,Int32,)` | Creates batches with padding to ensure uniform batch sizes. |
| `Sample(DataPipeline<>,IReadOnlyList<Double>,Int32,Nullable<Int32>)` | Samples elements with replacement using the given weights. |
| `ToAsyncPipeline(IStreamingDataLoader<,,>,Boolean,Nullable<Int32>)` | Creates an async DataPipeline from a streaming data loader. |
| `ToPipeline(IBatchIterable<>)` | Creates a DataPipeline from a batch iterable. |
| `ToPipeline(IEnumerable<>)` | Creates a DataPipeline from an enumerable. |
| `ToPipeline(IStreamingDataLoader<,,>,Boolean,Nullable<Int32>)` | Creates a DataPipeline from a streaming data loader. |
| `ToSamplePipeline(IStreamingDataLoader<,,>,Boolean,Nullable<Int32>)` | Creates a DataPipeline of individual samples from a streaming data loader. |
| `Window(DataPipeline<>,Int32,Nullable<Int32>)` | Applies window-based operations to the pipeline. |

