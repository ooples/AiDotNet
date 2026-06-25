---
title: "DataLoaderBase<T>"
description: "Abstract base class providing common functionality for all data loaders."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Data.Loaders`

Abstract base class providing common functionality for all data loaders.

## For Beginners

This class handles the "boring but important" stuff
that all data loaders need to do: tracking where you are in the data, resetting
to start over, and making sure data is loaded before you try to use it.

When you create a custom data loader, you extend one of the domain-specific base
classes (like GraphDataLoaderBase) which in turn extends this class, so you get
all this functionality for free.

## How It Works

DataLoaderBase implements shared functionality for all data loaders including:

- State management (loaded/unloaded)
- Iteration tracking (current index, progress)
- Reset functionality
- Thread-safe operations where needed

Domain-specific base classes (GraphDataLoaderBase, InputOutputDataLoaderBase)
extend this class with specialized functionality.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DataLoaderBase(Int32)` | Initializes a new instance of the DataLoaderBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchCount` |  |
| `BatchSize` | Gets or sets the batch size for iteration. |
| `CurrentBatchIndex` |  |
| `CurrentIndex` |  |
| `Description` |  |
| `IsLoaded` |  |
| `Name` |  |
| `Progress` |  |
| `TotalCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AdvanceBatchIndex` | Advances the batch index by one. |
| `AdvanceIndex(Int32)` | Advances the current index by the specified amount. |
| `EnsureLoaded` | Ensures data is loaded before operations that require it. |
| `LoadAsync(CancellationToken)` |  |
| `LoadDataCoreAsync(CancellationToken)` | Core data loading implementation to be provided by derived classes. |
| `OnReset` | Called after Reset() to allow derived classes to perform additional reset operations. |
| `Reset` |  |
| `Unload` |  |
| `UnloadDataCore` | Core data unloading implementation to be provided by derived classes. |

