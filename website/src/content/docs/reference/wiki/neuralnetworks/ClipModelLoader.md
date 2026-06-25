---
title: "ClipModelLoader"
description: "Loads CLIP models from HuggingFace Hub or local directories."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.NeuralNetworks`

Loads CLIP models from HuggingFace Hub or local directories.

## For Beginners

Instead of manually downloading files, this loader
automatically fetches CLIP models from HuggingFace Hub:

The model files are cached locally so subsequent loads are fast.

## How It Works

This loader handles downloading and caching CLIP model files from HuggingFace,
including ONNX model files for image and text encoders, tokenizer files,
and model configuration.

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearCache(String,String)` | Clears the cached model files. |
| `FromDirectory(String,ClipModelConfig)` | Loads a CLIP model from a local directory. |
| `FromPretrained(String,String)` | Loads a CLIP model from HuggingFace Hub synchronously. |
| `FromPretrainedAsync(String,String,IProgress<Double>,CancellationToken)` | Loads a CLIP model from HuggingFace Hub asynchronously. |
| `IsModelCached(String,String)` | Checks if a model is downloaded and cached locally. |
| `SanitizeModelIdForPath(String)` | Sanitizes a model ID for use as a directory name. |
| `ValidateAndCombinePath(String,String)` | Validates that a combined path does not escape the base directory (path traversal protection). |

## Fields

| Field | Summary |
|:-----|:--------|
| `KnownModels` | Known CLIP model configurations on HuggingFace Hub. |

