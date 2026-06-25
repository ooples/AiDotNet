---
title: "IDataLoader<T>"
description: "Base interface for all data loaders providing common data loading capabilities."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Base interface for all data loaders providing common data loading capabilities.

## For Beginners

Think of IDataLoader as the foundation that all data loaders build upon.

Just like all vehicles (cars, trucks, motorcycles) share common features (wheels, engine),
all data loaders share these common features:

- A name and description so you know what data it loads
- The ability to load and unload data
- The ability to track how much data there is and where you are in processing it

Specific types of data loaders (for images, graphs, text, etc.) add their own
specialized features on top of this foundation.

## How It Works

IDataLoader defines the foundation for all specialized data loaders in the system.
It provides:

- Basic metadata (name, description)
- Load/unload lifecycle management
- Reset capability for multi-epoch training
- Progress tracking through ICountable

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` | Gets a description of the dataset and its intended use. |
| `IsLoaded` | Gets whether the data has been loaded and is ready for iteration. |
| `Name` | Gets the human-readable name of this data loader. |

## Methods

| Method | Summary |
|:-----|:--------|
| `LoadAsync(CancellationToken)` | Loads the data asynchronously, preparing it for iteration. |
| `Unload` | Unloads the data and releases associated resources. |

