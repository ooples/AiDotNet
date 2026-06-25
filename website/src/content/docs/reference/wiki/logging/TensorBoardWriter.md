---
title: "TensorBoardWriter"
description: "Low-level TensorBoard event file writer."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Logging`

Low-level TensorBoard event file writer.

## For Beginners

TensorBoard is a visualization tool from TensorFlow.

This writer creates event files that TensorBoard can read and display.
It's like writing a diary in a specific format that TensorBoard knows
how to read and show as beautiful charts and graphs.

Event files contain:

- Scalar values (loss, accuracy over time)
- Histograms (weight distributions)
- Images (sample outputs, feature maps)
- Text (descriptions, annotations)
- Graphs (model architecture)

## How It Works

TensorBoard event files use a specific binary format consisting of records.
Each record contains: length (8 bytes), masked CRC of length (4 bytes),
data (variable), and masked CRC of data (4 bytes).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TensorBoardWriter(String,String)` | Creates a new TensorBoard event file writer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FilePath` | Gets the event file path. |
| `LogDir` | Gets the log directory path. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Dispose` | Releases resources. |
| `Flush` | Flushes pending writes to disk. |
| `WriteEmbedding(String,Single[0:,0:],String[],Int64)` | Writes an embedding with optional metadata and sprite. |
| `WriteHistogram(String,ReadOnlySpan<Single>,Int64)` | Writes a histogram summary from a tensor. |
| `WriteHistogram(String,Single[],Int64)` | Writes a histogram summary. |
| `WriteImage(String,Byte[],Int32,Int32,Int64)` | Writes an image summary. |
| `WriteImageRaw(String,Byte[],Int32,Int32,Int32,Int64)` | Writes raw image data (HWC format, values 0-255). |
| `WriteScalar(String,Single,Int64)` | Writes a scalar summary to the event file. |
| `WriteScalars(String,Dictionary<String,Single>,Int64)` | Writes multiple scalars as a group. |
| `WriteText(String,String,Int64)` | Writes text summary. |

