---
title: "LmdbDataset<T>"
description: "Provides read-only access to datasets stored in a custom binary key-value format inspired by LMDB (Lightning Memory-Mapped Database)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Formats`

Provides read-only access to datasets stored in a custom binary key-value format
inspired by LMDB (Lightning Memory-Mapped Database).

## How It Works

**Important:** This is NOT a native LMDB implementation and is NOT compatible with
LMDB databases created by other tools. It is a custom binary key-value store format
designed for efficient sequential and random access to ML training data within AiDotNet.
For native LMDB interop, use the LightningDB NuGet package.

The on-disk format is a simple binary key-value store:

- Header: [magic: 4 bytes "LMDB"] [version: 4 bytes] [numEntries: 4 bytes] [indexOffset: 8 bytes]
- Data section: sequential key-value pairs [keyLen: 4][key: bytes][valueLen: 4][value: bytes]
- Index section: [offset: 8 bytes] per entry, pointing to start of each key-value pair

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LmdbDataset(LmdbDatasetOptions)` | Opens an LMDB dataset from a directory containing a data.mdb file. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Count` | Gets the number of entries in the dataset. |
| `Keys` | Gets all keys in the dataset. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Dispose` |  |
| `Get(String)` | Gets a raw byte value by key. |
| `GetAsArray(String)` | Gets a value as a double array (assumes 8-byte doubles stored sequentially). |
| `GetAsString(String)` | Gets a value as a string. |
| `GetByIndex(Int32)` | Gets a value by integer index. |
| `WriteDataset(String,IReadOnlyList<KeyValuePair<String,Byte[]>>)` | Writes a dataset to LMDB format. |

