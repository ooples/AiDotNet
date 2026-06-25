---
title: "JsonlDataLoader<T>"
description: "A typed data loader facade that wraps `JsonlStreamingLoader` and implements `StreamingDataLoaderBase` for IDataLoader compliance."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Formats`

A typed data loader facade that wraps `JsonlStreamingLoader` and implements
`StreamingDataLoaderBase` for IDataLoader compliance.

## For Beginners

Use this when you want to load JSONL files and feed them into
`AiModelBuilder.ConfigureDataLoader()`. You provide a parser function that knows how
to convert each JSON record into tensors for training.

## How It Works

This loader eagerly reads all JSON objects from the underlying JSONL files during loading,
caches them in memory, and provides index-based access via a user-supplied parser delegate
that converts `JObject` records into typed tensor pairs.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `JsonlDataLoader(String[],Func<JObject,ValueTuple<Tensor<>,Tensor<>>>,Int32,String,String,Int32)` | Creates a new JsonlDataLoader. |

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

