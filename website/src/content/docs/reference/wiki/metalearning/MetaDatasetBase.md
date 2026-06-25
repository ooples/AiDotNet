---
title: "MetaDatasetBase<T, TInput, TOutput>"
description: "Abstract base class for meta-datasets that generate episodes on-the-fly."
section: "API Reference"
---

`Base Classes` · `AiDotNet.MetaLearning.Data`

Abstract base class for meta-datasets that generate episodes on-the-fly.
Subclasses provide the data; this base class handles episode construction and validation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MetaDatasetBase(Nullable<Int32>)` | Initializes the base meta-dataset with an optional seed. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassExampleCounts` |  |
| `Name` |  |
| `TotalClasses` |  |
| `TotalExamples` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `SampleEpisode(Int32,Int32,Int32)` |  |
| `SampleEpisodes(Int32,Int32,Int32,Int32)` |  |
| `SampleTaskCore(Int32,Int32,Int32)` | Core method that samples a single meta-learning task. |
| `SampleWithoutReplacement(Int32,Int32)` | Selects `count` random distinct integers from [0, `max`). |
| `SetSeed(Int32)` |  |
| `SupportsConfiguration(Int32,Int32,Int32)` |  |
| `ValidateConfiguration(Int32,Int32,Int32)` | Validates that the requested configuration is feasible for this dataset. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Numeric operations for type T. |
| `Rng` | Random number generator for sampling. |

