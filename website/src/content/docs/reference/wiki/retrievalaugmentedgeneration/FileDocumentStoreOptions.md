---
title: "FileDocumentStoreOptions"
description: "Configuration options for the file-based document store."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.RetrievalAugmentedGeneration.DocumentStores`

Configuration options for the file-based document store.

## For Beginners

These settings control how the file-based document store
operates, including where files are stored, HNSW index parameters, and WAL behavior.
The defaults work well for most use cases.

## Properties

| Property | Summary |
|:-----|:--------|
| `CompactionTombstoneRatio` | Gets or sets the ratio of tombstones to total documents that triggers automatic compaction. |
| `DirectoryPath` | Gets or sets the directory path where store files are written. |
| `FlushOnEveryWrite` | Gets or sets whether to flush data to disk after every write operation. |
| `HnswEfConstruction` | Gets or sets the HNSW efConstruction parameter (search depth during index building). |
| `HnswEfSearch` | Gets or sets the HNSW efSearch parameter (search depth during queries). |
| `HnswMaxConnections` | Gets or sets the HNSW M parameter (max connections per node). |
| `HnswSeed` | Gets or sets the random seed for the HNSW index. |
| `MaxWalSizeBytes` | Gets or sets the maximum WAL size in bytes before auto-flushing to main files. |
| `MinimumDocumentCountForCompaction` | Gets or sets the minimum total document count (live + tombstones) required before automatic compaction is considered. |
| `SyncWalWrites` | Gets or sets whether WAL writes should be synchronous (fsync on every write). |

