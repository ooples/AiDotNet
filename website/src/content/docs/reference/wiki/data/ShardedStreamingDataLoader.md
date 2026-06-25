---
title: "ShardedStreamingDataLoader<T>"
description: "A typed data loader facade that wraps `ShardedStreamingDataset` and implements `StreamingDataLoaderBase` for IDataLoader compliance."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Formats`

A typed data loader facade that wraps `ShardedStreamingDataset` and implements
`StreamingDataLoaderBase` for IDataLoader compliance.

## For Beginners

Use this when you want to load sharded binary datasets and
feed them into `AiModelBuilder.ConfigureDataLoader()`. You provide a parser function
that knows how to decode each binary record into tensors for training.

## How It Works

This loader eagerly reads all records from the underlying sharded binary files during loading,
caches them in memory, and provides index-based access via a user-supplied parser delegate
that converts raw `byte[]` records into typed tensor pairs.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ShardedStreamingDataLoader(String[],Func<Byte[],ValueTuple<Tensor<>,Tensor<>>>,Int32,ShardedStreamingDatasetOptions)` | Creates a new ShardedStreamingDataLoader. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SampleCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `LoadDataCoreAsync(CancellationToken)` |  |
| `ReadSampleAsync(Int32,CancellationToken)` |  |
| `UnloadDataCore` |  |

