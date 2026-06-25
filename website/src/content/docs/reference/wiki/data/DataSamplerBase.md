---
title: "DataSamplerBase"
description: "Base class for all data samplers providing common functionality."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Data.Sampling`

Base class for all data samplers providing common functionality.

## For Beginners

This base class handles the common plumbing that all
samplers need, like managing random number generators for reproducibility.
When creating a custom sampler, inherit from this class and override GetIndicesCore().

## How It Works

DataSamplerBase provides default implementations for common sampler operations
like random seed management and epoch callbacks. All concrete samplers should
inherit from this base class rather than implementing IDataSampler directly.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DataSamplerBase(Nullable<Int32>)` | Initializes a new instance of the DataSamplerBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Length` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateSequentialIndices(Int32)` | Creates a sequential array of indices from 0 to count-1. |
| `GetIndices` |  |
| `GetIndicesCore` | Core implementation for generating indices. |
| `OnEpochStart(Int32)` |  |
| `SetSeed(Int32)` |  |
| `ShuffleIndices(Int32[])` | Performs Fisher-Yates shuffle on an array of indices. |

## Fields

| Field | Summary |
|:-----|:--------|
| `CurrentEpoch` | The current epoch number (0-based). |
| `Random` | The random number generator used for sampling. |

