---
title: "TextLineDataset<T>"
description: "Streams text line-by-line from a file for language modeling tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Text`

Streams text line-by-line from a file for language modeling tasks.

## For Beginners

This loader is for large text files that you want to
process line by line. Each line becomes one sample in your dataset.

## How It Works

Reads a text file line by line, providing each line as a string sample.
Extends DataLoaderBase for lifecycle management while exposing line-reading
as an async enumerable.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TextLineDataset(String,Boolean,Int32)` | Creates a new TextLineDataset that reads from the specified file. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `FilePath` | Gets the file path being read. |
| `Name` |  |
| `TotalCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `LoadDataCoreAsync(CancellationToken)` |  |
| `ReadBatchesAsync(Nullable<Int32>,CancellationToken)` | Reads lines in batches. |
| `ReadLinesAsync(CancellationToken)` | Reads all lines from the text file. |
| `UnloadDataCore` |  |

