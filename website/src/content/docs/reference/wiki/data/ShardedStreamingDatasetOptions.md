---
title: "ShardedStreamingDatasetOptions"
description: "Configuration options for the sharded streaming dataset."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Data.Formats`

Configuration options for the sharded streaming dataset.

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxSamples` | Optional maximum number of samples to read per epoch. |
| `NumWorkers` | Reserved for future support of parallel shard reading. |
| `Seed` | Optional random seed for reproducible shuffling. |
| `ShuffleBufferSize` | Buffer size for within-shard shuffling. |
| `ShuffleShards` | Whether to shuffle shard order each epoch. |

