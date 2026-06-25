---
title: "MetaDatasetFormat<T, TInput, TOutput>"
description: "Supports the Meta-Dataset benchmark format (Triantafillou et al., 2020): a multi-domain evaluation protocol with variable-way variable-shot task sampling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Data`

Supports the Meta-Dataset benchmark format (Triantafillou et al., 2020): a multi-domain
evaluation protocol with variable-way variable-shot task sampling. Wraps multiple
`IMetaDataset` instances and samples episodes from a
randomly chosen domain, or from a specific domain on request.

## For Beginners

The Meta-Dataset benchmark tests how well a model generalizes
across multiple very different domains (e.g., birds, textures, aircraft, fungi). This class
lets you combine several datasets and sample tasks from any of them.

## How It Works

**Reference:** Meta-Dataset: A Dataset of Datasets for Learning to Learn
from Few Examples (Triantafillou et al., ICLR 2020).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MetaDatasetFormat(IReadOnlyList<IMetaDataset<,,>>,IReadOnlyList<String>,Nullable<Int32>)` | Creates a multi-domain meta-dataset from multiple individual datasets. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassExampleCounts` |  |
| `DomainCount` | Gets the number of domains in this multi-domain dataset. |
| `DomainNames` | Gets the domain names. |
| `Name` |  |
| `TotalClasses` |  |
| `TotalExamples` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `SampleEpisode(Int32,Int32,Int32)` |  |
| `SampleEpisodes(Int32,Int32,Int32,Int32)` |  |
| `SampleFromDomain(Int32,Int32,Int32,Int32)` | Samples an episode from a specific domain by index. |
| `SetSeed(Int32)` |  |
| `SupportsConfiguration(Int32,Int32,Int32)` |  |

