---
title: "OnnxModelDownloader"
description: "Downloads ONNX models from HuggingFace Hub and other repositories."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Onnx`

Downloads ONNX models from HuggingFace Hub and other repositories.

## For Beginners

Instead of manually downloading model files, use this class:

## How It Works

This class provides functionality to download pre-trained ONNX models from
HuggingFace Hub with automatic caching, progress reporting, and resume support.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OnnxModelDownloader(HttpClient,String,Boolean)` | Creates a new OnnxModelDownloader with a custom HttpClient. |
| `OnnxModelDownloader(String)` | Creates a new OnnxModelDownloader with default settings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearCache(String)` |  |
| `Dispose` | Disposes the downloader and the HttpClient if owned. |
| `Dispose(Boolean)` | Disposes managed and unmanaged resources. |
| `DownloadAsync(String,String,IProgress<Double>,CancellationToken)` |  |
| `DownloadFromUrlAsync(String,IProgress<Double>,CancellationToken)` | Downloads a model from a direct URL (not HuggingFace). |
| `DownloadMultipleAsync(String,IEnumerable<String>,IProgress<Double>,CancellationToken)` |  |
| `GetCacheSize` |  |
| `GetCachedFiles(String)` | Gets the files cached for a specific model. |
| `GetCachedPath(String,String)` |  |
| `ListCachedModels` | Lists all cached models. |

## Fields

| Field | Summary |
|:-----|:--------|
| `HuggingFaceBaseUrl` | The base URL for HuggingFace Hub. |

