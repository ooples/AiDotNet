---
title: "CsvStreamingDataLoader<T, TInput, TOutput>"
description: "A streaming data loader that reads from a CSV file line by line."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Loaders`

A streaming data loader that reads from a CSV file line by line.

## For Beginners

If you have a large CSV file (gigabytes of data), this loader
will read it row by row as needed during training.

Example:

## How It Works

CsvStreamingDataLoader reads a CSV file line by line without loading the entire file
into memory. This is ideal for large tabular datasets.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CsvStreamingDataLoader(String,Func<String,Int32,ValueTuple<,>>,Int32,Boolean,Int32,Int32)` | Initializes a new instance of the CsvStreamingDataLoader class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SampleCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSequentialBatches(Nullable<Int32>,Boolean)` | Iterates through the CSV file sequentially without loading all lines into memory. |
| `ReadSampleAsync(Int32,CancellationToken)` |  |
| `UnloadDataCore` |  |

