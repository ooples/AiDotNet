---
title: "IOnnxModelDownloader"
description: "Defines the contract for downloading ONNX models from remote sources."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for downloading ONNX models from remote sources.

## For Beginners

Instead of manually downloading model files,
you can use implementers of this interface to automatically fetch models:

## How It Works

This interface provides a way to download ONNX models from repositories
like HuggingFace Hub or ONNX Model Zoo. It supports progress reporting
and caching of downloaded models.

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearCache(String)` | Clears the local cache for a specific model or all models. |
| `DownloadAsync(String,String,IProgress<Double>,CancellationToken)` | Downloads an ONNX model from a remote repository. |
| `DownloadMultipleAsync(String,IEnumerable<String>,IProgress<Double>,CancellationToken)` | Downloads multiple ONNX files from a model repository. |
| `GetCacheSize` | Gets the total size of the local cache in bytes. |
| `GetCachedPath(String,String)` | Checks if a model is already cached locally. |

