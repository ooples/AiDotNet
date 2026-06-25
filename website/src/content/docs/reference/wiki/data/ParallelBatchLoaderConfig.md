---
title: "ParallelBatchLoaderConfig"
description: "Configuration for parallel batch loading."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Loaders`

Configuration for parallel batch loading.

## Properties

| Property | Summary |
|:-----|:--------|
| `NumWorkers` | Gets or sets the number of worker threads. |
| `PersistentWorkers` | Gets or sets whether to use persistent workers that stay alive between epochs. |
| `PinMemory` | Gets or sets whether to pin memory for faster CPU-to-GPU transfer. |
| `PrefetchCount` | Gets or sets the number of batches to prefetch. |
| `WorkerTimeoutMs` | Gets or sets the timeout for worker operations in milliseconds. |

