---
title: "WeightDownloader"
description: "Downloads and caches pre-trained model weights from remote URLs."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Weights`

Downloads and caches pre-trained model weights from remote URLs.

## For Beginners

Pre-trained weights are model parameters that have been
trained on large datasets (like COCO or ImageNet). Instead of training from scratch,
you can download these weights and either use them directly or fine-tune them on
your own data. This class handles downloading and caching these weight files.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WeightDownloader` | Creates a new weight downloader with the default cache directory. |
| `WeightDownloader(String)` | Creates a new weight downloader with a custom cache directory. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearCache` | Clears all cached weights. |
| `DownloadAsync(String,String,IProgress<Double>,CancellationToken)` | Downloads a weight file from a URL. |
| `DownloadIfNeededAsync(String,String,IProgress<Double>,CancellationToken)` | Downloads weights if not already cached. |
| `GetCachePath(String)` | Gets the cached path for a weight file. |
| `GetCacheSize` | Gets the total size of cached weights in bytes. |
| `GetDefaultCacheDirectory` | Gets the default cache directory for storing weights. |
| `IsCached(String)` | Checks if weights are already cached. |
| `RemoveFromCache(String)` | Removes a weight file from cache. |

