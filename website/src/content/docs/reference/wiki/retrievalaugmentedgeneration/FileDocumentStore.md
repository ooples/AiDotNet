---
title: "FileDocumentStore<T>"
description: "A durable, file-based vector document store with HNSW indexing and write-ahead logging."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.DocumentStores`

A durable, file-based vector document store with HNSW indexing and write-ahead logging.

## For Beginners

This is like a persistent library that survives application restarts.

Key features:

- Data is saved to disk files so it survives crashes and restarts
- Fast similarity search using HNSW graph index (same as the in-memory store)
- Write-ahead log ensures no data loss even during crashes
- Deleted documents are marked (not immediately removed) and cleaned up periodically

File layout on disk:

Best used for:

- Applications that need persistent vector search
- Medium datasets (up to ~1M documents depending on RAM)
- Single-process applications (not for multi-process concurrent access)
- Scenarios where you want an embedded vector database without external dependencies

**Thread-safety:** While `ConcurrentDictionary` is used
internally for document/metadata storage, HNSW index operations (AddVectors, RemoveVectors,
Search/Query) are **not** protected by internal locks. Concurrent mutations from multiple
threads are unsupported. Concurrent read-only/search operations may work depending on the
HNSW implementation but are not guaranteed. For safe in-process concurrency, serialize writes
with an external lock or implement internal synchronization around HNSW accesses.

## How It Works

FileDocumentStore persists documents to disk while maintaining an in-memory HNSW index
for O(log n) approximate nearest neighbor search. It uses a write-ahead log (WAL) for
crash recovery and tombstone-based soft deletes with periodic compaction.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FileDocumentStore(Int32,FileDocumentStoreOptions)` | Initializes a new FileDocumentStore, creating or loading from the specified directory. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DirectoryPath` | Gets the directory path where store files are located. |
| `DocumentCount` |  |
| `TombstoneCount` | Gets the number of tombstoned (soft-deleted) documents awaiting compaction. |
| `VectorDimension` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddBatchCore(IList<VectorDocument<>>)` | Core logic for batch-adding vector documents. |
| `AddCore(VectorDocument<>)` | Core logic for adding a single vector document. |
| `CheckAutoFlush` | Checks if auto-flush or auto-compaction should be triggered. |
| `Clear` | Removes all documents from the store and deletes all files. |
| `ClearWal` | Clears the WAL file after a successful flush. |
| `CloseWalWriter` | Closes the WAL writer. |
| `Compact` | Performs compaction: rebuilds store files removing tombstoned entries and reclaiming space. |
| `DeleteFileIfExists(String)` | Deletes a file if it exists, ignoring errors. |
| `Dispose` | Releases resources used by the store. |
| `Dispose(Boolean)` | Releases resources used by the store. |
| `DoubleArrayToVector(Double[])` | Converts a double array back to a Vector<T>. |
| `Flush` | Flushes all in-memory data to disk, creating a consistent snapshot. |
| `GetAllCore` | Core logic for retrieving all documents. |
| `GetByIdCore(String)` | Core logic for retrieving a document by ID. |
| `GetSimilarCore(Vector<>,Int32,Dictionary<String,Object>)` | Core logic for similarity search using HNSW index with optional metadata filtering. |
| `LoadFromDisk` | Loads all data from disk files into memory. |
| `OpenWalWriter` | Opens the WAL writer for append operations. |
| `PersistToDisk` | Persists all in-memory data to disk files. |
| `ReadDocuments` | Reads all documents from the JSON file. |
| `ReadMetadata` | Reads the store metadata header file. |
| `ReadVectors(Int32,Int32)` | Reads vectors from binary file. |
| `RemoveCore(String)` | Core logic for removing a document. |
| `ReplayWal` | Replays WAL entries to recover any uncommitted operations. |
| `ThrowIfDisposed` | Throws ObjectDisposedException if the store has been disposed. |
| `VectorToDoubleArray(Vector<>)` | Converts a Vector<T> to a double array for WAL serialization. |
| `WriteDocuments(KeyValuePair<String,VectorDocument<>>[])` | Writes all documents (content + metadata) as JSON. |
| `WriteHnswGraph` | Writes the HNSW graph marker file. |
| `WriteMetadata(Int32)` | Writes the store metadata header file. |
| `WriteVectors(KeyValuePair<String,VectorDocument<>>[])` | Writes all vectors as compact binary data. |
| `WriteWalEntry(FileDocumentStore<>.WalEntry)` | Writes a single entry to the WAL. |

