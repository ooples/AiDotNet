---
title: "BTreeIndex"
description: "Simple file-based index for mapping string keys to file offsets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Graph`

Simple file-based index for mapping string keys to file offsets.

## For Beginners

Think of this like a book's index at the back.

Without an index:

- To find "photosynthesis", you'd read every page from start to finish
- Very slow for large books (or large data files)

With an index:

- Look up "photosynthesis" → find it's on page 157
- Jump directly to page 157
- Much faster!

This class does the same for graph data:

- Key: "node_alice_001"
- Value: byte offset 45678 in nodes.dat file
- We can jump directly to byte 45678 to read Alice's data

The index itself is stored in a file so it survives application restarts.

## How It Works

This class provides a persistent index structure that maps string keys (e.g., node IDs)
to byte offsets in data files. The index is stored on disk and reloaded on restart,
enabling fast lookups without scanning entire data files.

**Implementation Note:** This is a simplified index using a sorted dictionary.
For production systems with millions of entries, consider implementing a true
B-Tree structure with splitting/merging nodes, or use an embedded database like
SQLite or LevelDB.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BTreeIndex(String)` | Initializes a new instance of the `BTreeIndex` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Count` | Gets the number of entries in the index. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Add(String,Int64)` | Adds or updates a key-offset mapping in the index. |
| `Clear` | Removes all entries from the index. |
| `Contains(String)` | Checks if a key exists in the index. |
| `Dispose` | Disposes the index, ensuring all changes are saved to disk. |
| `Dispose(Boolean)` | Releases resources used by the index. |
| `Flush` | Saves the index to disk if it has been modified. |
| `Get(String)` | Retrieves the file offset associated with a key. |
| `GetAllKeys` | Gets all keys in the index. |
| `LoadFromDisk` | Loads the index from disk if it exists. |
| `Remove(String)` | Removes a key from the index. |

