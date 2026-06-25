---
title: "HuggingFaceModelLoader<T>"
description: "Downloads and caches models from HuggingFace Hub."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ModelLoading`

Downloads and caches models from HuggingFace Hub.

## For Beginners

HuggingFace Hub is like a library of pretrained AI models.

Instead of training models yourself (which requires huge amounts of data and compute),
you can download models that others have already trained. This class handles:

1. Downloading model files from HuggingFace
2. Caching them locally so you don't re-download every time
3. Loading the weights into your model

Example usage:
```cs
var loader = new HuggingFaceModelLoader<float>();

// Download and cache a pretrained VAE
var files = await loader.DownloadModelAsync("stabilityai/sd-vae-ft-mse");

// Load weights into your model
var vae = new VAEEncoder<float>();
loader.LoadWeights(vae, files["diffusion_pytorch_model.safetensors"]);
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HuggingFaceModelLoader(String,String,Boolean)` | Initializes a new instance of the HuggingFaceModelLoader class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearAllCache` | Clears all cached models. |
| `ClearCache(String)` | Clears the cache for a specific model. |
| `ComputeHash(String)` | Computes a short hash for cache directory naming. |
| `CopyWithProgressAsync(Stream,Stream,Int64,String,CancellationToken)` | Copies stream with progress reporting. |
| `DownloadAndLoadAsync(IWeightLoadable<>,String,String,Func<String,String>,String,Boolean,CancellationToken)` | Downloads a model and loads its weights in one operation. |
| `DownloadFileAsync(String,String,String,CancellationToken)` | Downloads a single file from HuggingFace Hub. |
| `DownloadModelAsync(String,String,IEnumerable<String>,CancellationToken)` | Downloads a model from HuggingFace Hub. |
| `GetCachePath(String,String)` | Gets the local cache path for a repository. |
| `GetDefaultCacheDir` | Gets the default cache directory. |
| `GetRepoFilesAsync(String,String,CancellationToken)` | Gets the list of files in a HuggingFace repository. |
| `IsCached(String,String,String)` | Checks if a model is already cached locally. |
| `LoadWeights(IWeightLoadable<>,String,Func<String,String>,Boolean)` | Loads weights from a downloaded SafeTensors file into a model. |
| `LoadWeights(IWeightLoadable<>,String,WeightMapping,Boolean)` | Loads weights using a WeightMapping instance. |
| `MatchesAnyPattern(String,List<String>)` | Checks if a file name matches any of the given patterns. |
| `MatchesWildcard(String,String)` | Simple wildcard matching (supports * only). |

## Fields

| Field | Summary |
|:-----|:--------|
| `HF_API_URL` | Default HuggingFace Hub API URL. |
| `_apiToken` | Optional HuggingFace API token for accessing private/gated models. |
| `_cacheDir` | Local cache directory for downloaded models. |
| `_httpClient` | HTTP client for API requests. |
| `_pretrainedLoader` | Pretrained model loader for applying weights. |
| `_safeTensorsLoader` | SafeTensors loader for loading weights. |
| `_verbose` | Whether to log download progress. |

