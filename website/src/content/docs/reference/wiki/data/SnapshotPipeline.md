---
title: "SnapshotPipeline<T>"
description: "Persists an entire processed pipeline to disk for fast reload across epochs, with automatic invalidation when source data or pipeline configuration changes."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Pipeline`

Persists an entire processed pipeline to disk for fast reload across epochs,
with automatic invalidation when source data or pipeline configuration changes.

## For Beginners

Imagine you have a pipeline that reads images, resizes them,
and normalizes pixel values. On the first run, this is slow. SnapshotPipeline saves the
processed results so that the second run is instant:

## How It Works

Inspired by TensorFlow's tf.data snapshot pattern. After the first epoch processes
data through expensive transforms (decode, resize, augment, tokenize), the entire
output is saved to disk. Subsequent epochs load the cached result directly,
skipping all preprocessing.

Cache invalidation is automatic: the pipeline configuration and source data metadata
are hashed, and the cache is rebuilt when the hash changes.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SnapshotPipeline(DataPipeline<Tensor<>>,DiskCacheOptions,String)` | Creates a new snapshot pipeline. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ActiveCacheDirectory` | Gets the path to the active cache directory, or null if no cache exists. |
| `IsCacheValid` | Gets whether the cache is currently valid and up-to-date. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CleanupCache` | Cleans up old cache entries across all pipeline caches based on the eviction policy. |
| `GetCacheInfo` | Gets information about the current cache state. |
| `GetCachedPipeline(CancellationToken)` | Gets a pipeline that reads from cache if available, or processes and caches the source pipeline. |
| `InvalidateCache` | Invalidates and removes the cache. |
| `RebuildCache(CancellationToken)` | Forces a rebuild of the cache, even if a valid cache exists. |

