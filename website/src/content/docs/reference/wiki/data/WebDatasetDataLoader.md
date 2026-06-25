---
title: "WebDatasetDataLoader<T>"
description: "A typed data loader facade that wraps `WebDataset` and implements `StreamingDataLoaderBase` for IDataLoader compliance."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Formats`

A typed data loader facade that wraps `WebDataset` and implements
`StreamingDataLoaderBase` for IDataLoader compliance.

## For Beginners

Use this when you want to load WebDataset TAR archives and
feed them into `AiModelBuilder.ConfigureDataLoader()`. You provide a parser function
that knows how to decode the raw bytes (images, labels, etc.) into tensors.

## How It Works

This loader eagerly reads all samples from the underlying TAR archives during loading,
caches them in memory, and provides index-based access via a user-supplied parser delegate
that converts raw `Dictionary<string, byte[]>` samples into typed tensor pairs.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WebDatasetDataLoader(String[],Func<Dictionary<String,Byte[]>,ValueTuple<Tensor<>,Tensor<>>>,Int32,WebDatasetOptions)` | Creates a new WebDatasetDataLoader. |

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

