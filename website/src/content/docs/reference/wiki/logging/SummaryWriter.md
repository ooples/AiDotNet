---
title: "SummaryWriter"
description: "PyTorch-compatible SummaryWriter for TensorBoard logging."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Logging`

PyTorch-compatible SummaryWriter for TensorBoard logging.

## For Beginners

This is your interface to TensorBoard visualization.

During training, you use this writer to record:

- Loss values at each step (add_scalar)
- Model weight distributions (add_histogram)
- Sample outputs or feature maps (add_image)
- Model structure (add_graph)

Then you can visualize all this in TensorBoard by running:
tensorboard --logdir=your_log_directory

Example usage:

## How It Works

This class provides an API similar to PyTorch's torch.utils.tensorboard.SummaryWriter,
making it easy to log training metrics, model weights, images, and more.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SummaryWriter(String,String,Int32,Int32,Int32,String)` | Creates a new SummaryWriter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultStep` | Gets the current default step number. |
| `LogDir` | Gets the log directory path. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddCustomScalar(String,Single,Nullable<Int64>)` | Adds a custom scalar with layout. |
| `AddEmbedding(String,Single[0:,0:],String[],Single[0:,0:,0:,0:],Nullable<Int64>)` | Adds an embedding with optional metadata and labels. |
| `AddHistogram(String,ReadOnlySpan<Single>,Nullable<Int64>,Int32)` | Adds a histogram from a span of values. |
| `AddHistogram(String,Single[0:,0:],Nullable<Int64>,Int32)` | Adds a histogram from a 2D array (flattened). |
| `AddHistogram(String,Single[],Nullable<Int64>,Int32)` | Adds a histogram of values. |
| `AddHparams(Dictionary<String,Object>,Dictionary<String,Single>,Dictionary<String,Object[]>)` | Adds hyperparameters and associated metrics. |
| `AddImage(String,Single[0:,0:,0:],Nullable<Int64>,String)` | Adds an image to the summary. |
| `AddImageRaw(String,Byte[],Int32,Int32,Int32,Nullable<Int64>)` | Adds an image from raw pixel data. |
| `AddImages(String,Single[0:,0:,0:,0:],Nullable<Int64>,Int32,Int32,Boolean)` | Adds a grid of images. |
| `AddPrCurve(String,Int32[],Single[],Nullable<Int64>,Int32)` | Adds a PR curve for binary classification evaluation. |
| `AddScalar(String,Double,Nullable<Int64>)` | Adds a scalar value (double precision). |
| `AddScalar(String,Single,Nullable<Int64>)` | Adds a scalar value to the summary. |
| `AddScalars(String,Dictionary<String,Single>,Nullable<Int64>)` | Adds multiple scalars under a main tag. |
| `AddText(String,String,Nullable<Int64>)` | Adds text to the summary. |
| `Close` | Closes the writer (alias for Dispose). |
| `Dispose` | Releases resources and closes the writer. |
| `Flush` | Flushes pending writes to disk. |
| `LogTrainingStep(Single,Nullable<Single>,Nullable<Single>,Nullable<Int64>)` | Logs training metrics at the current step. |
| `LogValidationStep(Single,Nullable<Single>,Nullable<Int64>)` | Logs validation metrics. |
| `LogWeights(String,Single[],Single[],Nullable<Int64>)` | Logs model weight statistics. |

