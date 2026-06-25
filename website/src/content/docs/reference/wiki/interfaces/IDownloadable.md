---
title: "IDownloadable"
description: "Defines capability to automatically download and cache datasets."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines capability to automatically download and cache datasets.

## For Beginners

Many standard datasets (MNIST, CIFAR, Cora, etc.) are available online.
Instead of manually downloading and extracting files, the data loader can do it for you
automatically and remember where the files are stored.

## How It Works

Data loaders that implement this interface can fetch datasets from remote sources
and cache them locally, making it easy to use standard benchmark datasets
without manual setup.

## Properties

| Property | Summary |
|:-----|:--------|
| `CachePath` | Gets the local path where the dataset is cached. |
| `DownloadUrls` | Gets the URLs where the dataset can be downloaded from. |
| `IsDownloaded` | Gets whether the dataset has been downloaded and is available locally. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearCache` | Deletes the locally cached dataset files. |
| `DownloadAsync(Boolean,IProgress<Double>,CancellationToken)` | Downloads the dataset asynchronously if not already cached. |

