---
title: "AutoTokenizer"
description: "AutoTokenizer provides HuggingFace-style automatic tokenizer loading."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Tokenization.HuggingFace`

AutoTokenizer provides HuggingFace-style automatic tokenizer loading.
This class automatically detects and loads the appropriate tokenizer type
based on the model configuration.

## How It Works

Usage mirrors the HuggingFace transformers library:

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearCache(String,String)` | Clears the cache for a specific model or all models. |
| `FromPretrained(String,String)` | Loads a tokenizer from a pretrained model name or path. |
| `FromPretrainedAsync(String,String)` | Asynchronously loads a tokenizer from a pretrained model name or path. |
| `GetDefaultCacheDir` | Gets the default cache directory for tokenizer files. |
| `IsCached(String,String)` | Checks if a tokenizer is cached locally. |
| `ListCachedModels(String)` | Lists all cached tokenizer models. |

