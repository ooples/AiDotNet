---
title: "FileStreamingDataLoader<T, TInput, TOutput>"
description: "A streaming data loader that reads from files in a directory."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Loaders`

A streaming data loader that reads from files in a directory.

## For Beginners

If you have a folder full of images with labels in the filename
or a separate label file, this loader will read them one by one as needed.

Example:

## How It Works

FileStreamingDataLoader automatically discovers files in a directory and streams them
during training. This is ideal for image datasets where each file is a sample.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FileStreamingDataLoader(String,String,Func<String,CancellationToken,Task<ValueTuple<,>>>,Int32,SearchOption,Int32,Int32)` | Initializes a new instance of the FileStreamingDataLoader class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FilePaths` | Gets all file paths in the dataset. |
| `Name` |  |
| `SampleCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ReadSampleAsync(Int32,CancellationToken)` |  |

